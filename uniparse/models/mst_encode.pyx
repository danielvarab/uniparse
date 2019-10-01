"""MST feature extractor."""

# This implementation utilizes code and logic found in the project "Beta" by Marco Kuhlmann, licensed under CC 4.0. 
# https://creativecommons.org/licenses/by/4.0/
#
# Specifically this code adapts the bit shifting strategy and features found in EdgeFeaturizer.java as a rewrite in 
# Cython.
#
# The Beta project may be found at https://github.com/liu-nlp/beta
#
# The original bitshifting + feature extractor code may be found at
#     https://github.com/liu-nlp/beta/blob/master/src/main/java/se/liu/ida/nlp/beta/EdgeFeaturizer.java.


# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

cimport numpy as np
import numpy as np

# c++ imports
from cython.operator cimport postincrement as inc

from libcpp.unordered_map cimport unordered_map as map
from libcpp.vector cimport vector


cdef:
    np.uint64_t UPOS_OFFSET = 6 # ptb equires 50 labels, which can be represented in 6 bits
    np.uint64_t FORM_OFFSET = 20 # english universal dependencies requires 15 (15666 words)
    np.uint64_t EOS = 4
    np.uint64_t BOS = 3
    np.uint64_t MID = 5
    
    int TMP_00 = 0
    int TMP_01 = 1
    int TMP_02 = 2
    int TMP_03 = 3
    int TMP_04 = 4
    int TMP_05 = 5
    int TMP_06 = 6
    int TMP_07 = 7
    int TMP_08 = 8
    int TMP_09 = 9
    int TMP_10 = 10
    int TMP_11 = 11
    int TMP_12 = 12
    int TMP_13 = 13
    int TMP_14 = 14
    int TMP_15 = 15
    int TMP_16 = 16
    int TMP_17 = 17
    int TMP_18 = 18
    int TMP_19 = 19
    int TMP_20 = 20
    int TMP_21 = 21
    int TMP_22 = 22
    int TMP_23 = 23
    int TMP_24 = 24
    int TMP_25 = 25
    int TMP_26 = 26
    int TMP_27 = 27
    int TMP_28 = 28
    int TMP_29 = 29
    int TMP_30 = 30
    int TMP_31 = 31
    int TMP_32 = 32

cdef inline np.uint64_t makePair(int b1, int b2) nogil:
    return b2 << 1 | b1

cdef inline int encode_2(int f1, int f2) nogil:
    value = (f1 << FORM_OFFSET) | f2
    #return value
    return get_if_contained(value)
    
cdef inline int encode_3(int f1, int f2, int f3) nogil:
    value = (f1 << FORM_OFFSET | f2) << FORM_OFFSET | f3
    #return value
    return get_if_contained(value)
    
cdef inline int encode_4(int f1, int f2, int f3, int f4) nogil:
    value = ((f1 << FORM_OFFSET | f2) << FORM_OFFSET | f3) << UPOS_OFFSET | f4
    #return value
    return get_if_contained(value)

cdef inline int encode_5(int f1, int f2, int f3, int f4, int f5) nogil:
    value = (((f1 << FORM_OFFSET | f2) << FORM_OFFSET | f3) << UPOS_OFFSET | f4) << UPOS_OFFSET | f5
    #return value
    return get_if_contained(value)

cdef inline int encode_6(int f1, int f2, int f3, int f4, int f5, int f6) nogil:
    value = ((((f1 << FORM_OFFSET | f2) << FORM_OFFSET | f3) << UPOS_OFFSET | f4) << UPOS_OFFSET | f5) << UPOS_OFFSET | f6
    #return value
    return get_if_contained(value)


cdef int feature_index
cdef int mapper_locked = False
cdef map[int,int] mapper
cdef inline int get_if_contained(int value) nogil:
    cdef int tmp
    global feature_index
    if mapper.find(value) == mapper.end() and mapper_locked:
        return 0
    if mapper.find(value) == mapper.end() and not mapper_locked:
        tmp = feature_index
        feature_index += 1
        mapper[value] = tmp
        return tmp
    else:
        return mapper[value]


cdef class BetaEncodeHandler:
    def lock_feature_space(self):
        global mapper_locked
        mapper_locked = True


    def unlock_feature_space(self):
        global mapper_locked
        mapper_locked = False

    def __call__(self, np.uint64_t[:] forms, np.uint64_t[:] upos, np.int64_t[:] target_arcs=None, np.int64_t[:] target_rels=None):
        cdef int MAX_EDGE_FEATURE_COUNT = 370 # 366 is max in training
        cdef int MAX_RELS_FEATURE_COUNT = 28  # in our rel encoder there is only 14 features per element

        # indexes
        cdef int i
        cdef int j
        cdef int m
        cdef int h
        cdef int b

        # cdef int batch_size = forms.shape[0]
        cdef int n = forms.shape[0]

        # TODO: these can be native arrays
        cdef int label_count = 50
        cdef np.uint64_t[:] pred_t = np.zeros([n], dtype=np.uint64)
        cdef np.uint64_t[:] succ_t = np.zeros([n], dtype=np.uint64)

        cdef np.ndarray[np.uint64_t, ndim=3] arc_features = np.zeros([n, n, MAX_EDGE_FEATURE_COUNT], dtype=np.uint64)
        cdef np.ndarray[np.uint64_t, ndim=5] rel_features = np.zeros([n, label_count, 2, 2, MAX_RELS_FEATURE_COUNT], dtype=np.uint64)

        # for b in range(batch_size):
        pred_t[0] = BOS
        succ_t[n - 1] = EOS
        for i in range(1,n):
            pred_t[i] = upos[i - 1]

        for i in range(n-2, 0, -1):
            succ_t[i] = upos[i + 1]

        # values 
        cdef np.uint64_t _word
        cdef np.uint64_t _upos
        cdef np.uint64_t _pred_upos
        cdef np.uint64_t _succ_upos

        # for b in range(batch_size):
        for i in range(n):
            # encode labels
            for label in range(label_count):
                if target_rels is not None and target_rels[i] != label:
                    continue
                else:
                    _word = forms[i]
                    _upos = upos[i]
                    _pred_upos = pred_t[i]
                    _succ_upos = succ_t[i]

                    self.encode_rel(i, _word, _upos, _pred_upos, _succ_upos, label, isRA=False, isTarget=False, out=rel_features)
                    self.encode_rel(i, _word, _upos, _pred_upos, _succ_upos, label, isRA=True, isTarget=False, out=rel_features)

                    self.encode_rel(i, _word, _upos, _pred_upos, _succ_upos, label, isRA=False, isTarget=True, out=rel_features)
                    self.encode_rel(i, _word, _upos, _pred_upos, _succ_upos, label, isRA=True, isTarget=True, out=rel_features)

            # encode edge
            for j in range(n):
                if i == j:
                    continue
                elif target_arcs is not None and target_arcs[j] != i:
                    continue
                else:
                    self.encode_edge(i, j, forms, upos, pred_t, succ_t, arc_features)

        return arc_features, rel_features

    cpdef np.ndarray[np.uint64_t, ndim=3] convert_linear_scan(self, np.uint64_t[:,:,:,:,:,:] input_tensor):
        cdef int batch_size = input_tensor.shape[0]
        cdef int n = input_tensor.shape[1]
        cdef int label_count = input_tensor.shape[2]

        cdef np.ndarray[np.uint64_t, ndim=4] out_tensor = np.zeros((n, n, label_count, 28*2), dtype=np.uint64)

        cdef int b, head, modifier
        cdef int fi
        cdef int half = 28
        cdef int isRA
        cdef int label
        for b in range(batch_size):
            for head in range(n):
                for modifier in range(n):
                    isRA = head < modifier
                    for label in range(label_count):
                        for fi in range(28):
                            out_tensor[b, head, modifier, label, fi] = input_tensor[b, head, label, isRA, 0, fi]
                            out_tensor[b, head, modifier, label, half+fi] = input_tensor[b, modifier, label, isRA, 1, fi]
                            fi += 1

        return out_tensor

    cdef inline void encode_edge(self,
        int i, int j,
        np.uint64_t[:] forms,
        np.uint64_t[:] upos,
        np.uint64_t[:] pred_t,
        np.uint64_t[:] succ_t,
        np.uint64_t[:,:,:] out) nogil:

        cdef np.uint64_t fst_pred_t
        cdef np.uint64_t snd_succ_t
        cdef np.uint64_t fst_succ_t
        cdef np.uint64_t snd_pred_t
        
        cdef np.uint64_t fst_t
        cdef np.uint64_t snd_t
        cdef np.uint64_t mid_t

        cdef np.uint64_t fst
        cdef np.uint64_t snd
        cdef np.uint64_t isRA
        cdef np.uint64_t attDist
        cdef np.uint64_t distBool

        cdef np.uint64_t src_w
        cdef np.uint64_t src_t
        cdef np.uint64_t tgt_w
        cdef np.uint64_t tgt_t
        
        cdef int fi = 0 # feature index

        # TODO: swap these to make it head-major?
        fst = i
        snd = j

        fst, snd, isRA = (fst,snd,True) if i < j else (snd,fst,False)
        dist = snd-fst
        
        distBool = 0
        if(dist > 1):
            distBool = 1
        elif(dist > 2):
            distBool = 2
        elif(dist > 3):
            distBool = 3
        elif(dist > 4):
            distBool = 4
        elif(dist > 5):
            distBool = 5
        elif(dist > 10):
            distBool = 10
        attDist = distBool << 1 | (1 if isRA else 0)

        fst_t = forms[fst]
        snd_t = forms[snd]

        fst_pred_t = pred_t[fst]
        snd_succ_t = succ_t[snd]
        fst_succ_t = succ_t[fst] if fst < snd - 1 else MID
        snd_pred_t = pred_t[snd] if snd > fst + 1 else MID

        for mid in range(fst+1, snd):
            mid_t = upos[mid]
            out[i,j,inc(fi)] = encode_4(fst_t, snd_t, mid_t, TMP_00)
            out[i,j,inc(fi)] = encode_5(fst_t, snd_t, mid_t, attDist, TMP_00)

        out[i,j,inc(fi)] = encode_4(fst_pred_t, fst_t, snd_t, TMP_01)
        out[i,j,inc(fi)] = encode_5(fst_pred_t, fst_t, snd_t, attDist, TMP_01)

        out[i,j,inc(fi)] = encode_5(fst_pred_t, fst_t, snd_t, snd_succ_t, TMP_02)
        out[i,j,inc(fi)] = encode_6(fst_pred_t, fst_t, snd_t, snd_succ_t, attDist, TMP_02)

        out[i,j,inc(fi)] = encode_4(fst_pred_t, snd_t, snd_succ_t, TMP_03)
        out[i,j,inc(fi)] = encode_5(fst_pred_t, snd_t, snd_succ_t, attDist, TMP_03)

        out[i,j,inc(fi)] = encode_4(fst_pred_t, fst_t, snd_succ_t, TMP_04)
        out[i,j,inc(fi)] = encode_5(fst_pred_t, fst_t, snd_succ_t, attDist, TMP_04)

        out[i,j,inc(fi)] = encode_4(fst_t, snd_t, snd_succ_t, TMP_05)
        out[i,j,inc(fi)] = encode_5(fst_t, snd_t, snd_succ_t, attDist, TMP_05)

        out[i,j,inc(fi)] = encode_4(fst_t, fst_succ_t, snd_pred_t, TMP_06)
        out[i,j,inc(fi)] = encode_5(fst_t, fst_succ_t, snd_pred_t, attDist, TMP_06)

        out[i,j,inc(fi)] = encode_5(fst_t, fst_succ_t, snd_pred_t, snd_t, TMP_07)
        out[i,j,inc(fi)] = encode_6(fst_t, fst_succ_t, snd_pred_t, snd_t, attDist, TMP_07)

        out[i,j,inc(fi)] = encode_4(fst_t, fst_succ_t, snd_t, TMP_08)
        out[i,j,inc(fi)] = encode_5(fst_t, fst_succ_t, snd_t, attDist, TMP_08)

        out[i,j,inc(fi)] = encode_4(fst_t, snd_pred_t, snd_t, TMP_09)
        out[i,j,inc(fi)] = encode_5(fst_t, snd_pred_t, snd_t, attDist, TMP_09)

        out[i,j,inc(fi)] = encode_4(fst_succ_t, snd_pred_t, snd_t, TMP_10)
        out[i,j,inc(fi)] = encode_5(fst_succ_t, snd_pred_t, snd_t, attDist, TMP_10)

        out[i,j,inc(fi)] = encode_4(fst_t, snd_pred_t, snd_t, TMP_11)
        out[i,j,inc(fi)] = encode_5(fst_t, snd_pred_t, snd_t, attDist, TMP_11)

        out[i,j,inc(fi)] = encode_5(fst_t, fst_succ_t, snd_t, snd_succ_t, TMP_12)
        out[i,j,inc(fi)] = encode_6(fst_t, fst_succ_t, snd_t, snd_succ_t, attDist, TMP_12)

        src = fst
        tgt = snd

        src_w = forms[src]
        src_t = upos[src]
        tgt_w = forms[tgt]
        tgt_t = upos[tgt]

        out[i,j,inc(fi)] = encode_2(src_w, TMP_13)
        out[i,j,inc(fi)] = encode_3(src_w, attDist, TMP_13)

        out[i,j,inc(fi)] = encode_3(src_w, src_t, TMP_14)
        out[i,j,inc(fi)] = encode_4(src_w, src_t, attDist, TMP_14)

        out[i,j,inc(fi)] = encode_4(src_w, src_t, tgt_t, TMP_15)
        out[i,j,inc(fi)] = encode_5(src_w, src_t, tgt_t, attDist, TMP_15)

        out[i,j,inc(fi)] = encode_5(src_w, src_t, tgt_t, tgt_w, TMP_16)
        out[i,j,inc(fi)] = encode_6(src_w, src_t, tgt_t, tgt_w, attDist, TMP_16)

        out[i,j,inc(fi)] = encode_3(src_w, tgt_w, TMP_17)
        out[i,j,inc(fi)] = encode_4(src_w, tgt_w, attDist, TMP_17)

        out[i,j,inc(fi)] = encode_3(src_w, tgt_t, TMP_18)
        out[i,j,inc(fi)] = encode_4(src_w, tgt_t, attDist, TMP_18)

        out[i,j,inc(fi)] = encode_3(src_t, tgt_w, TMP_19)
        out[i,j,inc(fi)] = encode_4(src_t, tgt_w, attDist, TMP_19)

        out[i,j,inc(fi)] = encode_4(src_t, tgt_w, tgt_t, TMP_20)
        out[i,j,inc(fi)] = encode_5(src_t, tgt_w, tgt_t, attDist, TMP_20)

        out[i,j,inc(fi)] = encode_3(src_t, tgt_t, TMP_21)
        out[i,j,inc(fi)] = encode_4(src_t, tgt_t, attDist, TMP_21)

        out[i,j,inc(fi)] = encode_3(tgt_w, tgt_t, TMP_22)
        out[i,j,inc(fi)] = encode_4(tgt_w, tgt_t, attDist, TMP_22)

        out[i,j,inc(fi)] = encode_2(src_t, TMP_23)
        out[i,j,inc(fi)] = encode_3(src_t, attDist, TMP_23)

        out[i,j,inc(fi)] = encode_2(tgt_w, TMP_24)
        out[i,j,inc(fi)] = encode_3(tgt_w, attDist, TMP_24)
        
        out[i,j,inc(fi)] = encode_2(tgt_t, TMP_25)
        out[i,j,inc(fi)] = encode_3(tgt_t, attDist, TMP_25)
    
    cdef inline void encode_rel(self, 
        int i,
        int node_w,
        int node_t,
        int node_pred_t,
        int node_succ_t,
        int label,
        int isRA,
        int isTarget,
        np.uint64_t[:,:,:,:,:] out) nogil:

        cdef int index = 0

        suffix = makePair(isRA, isTarget) << 1 | 1

        out[i,label,isRA,isTarget,inc(index)] = encode_2(label, TMP_26)
        out[i,label,isRA,isTarget,inc(index)] = encode_3(label, suffix, TMP_26)

        out[i,label,isRA,isTarget,inc(index)] = encode_4(node_w, node_t, label, TMP_27)
        out[i,label,isRA,isTarget,inc(index)] = encode_5(node_w, node_t, label, suffix, TMP_27)

        out[i,label,isRA,isTarget,inc(index)] = encode_3(node_t, label, TMP_28)
        out[i,label,isRA,isTarget,inc(index)] = encode_4(node_t, label, suffix, TMP_28)

        out[i,label,isRA,isTarget,inc(index)] = encode_4(node_pred_t, node_t, label, TMP_29)
        out[i,label,isRA,isTarget,inc(index)] = encode_5(node_pred_t, node_t, label, suffix, TMP_29)

        out[i,label,isRA,isTarget,inc(index)] = encode_4(node_t, node_succ_t, label, TMP_30)
        out[i,label,isRA,isTarget,inc(index)] = encode_5(node_t, node_succ_t, label, suffix, TMP_30)

        out[i,label,isRA,isTarget,inc(index)] = encode_5(node_pred_t, node_t, node_succ_t, label, TMP_31)
        out[i,label,isRA,isTarget,inc(index)] = encode_6(node_pred_t, node_t, node_succ_t, label, suffix, TMP_31)
        
        out[i,label,isRA,isTarget,inc(index)] = encode_3(node_w, label, TMP_32)
        out[i,label,isRA,isTarget,inc(index)] = encode_4(node_w, label, suffix, TMP_32)