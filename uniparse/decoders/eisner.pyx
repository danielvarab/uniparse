cimport numpy as np
from numpy.math cimport INFINITY
cimport cython

import numpy as np

DTYPE = np.float64
BACKTRACK_DTYPE = np.int

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t BACKTRACK_DTYPE_t


@cython.boundscheck(False)
def Eisner(DTYPE_t[:,:] scores, BACKTRACK_DTYPE_t[:] gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    cdef int nr = scores.shape[0]
    cdef int nc = scores.shape[1]


    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    cdef int k,s,t,m,i,h

    cdef int N = nr - 1 # Number of words (excluding root).

    # Initialize CKY table.
    cdef np.ndarray[DTYPE_t, ndim=3] complete   = np.zeros([N+1, N+1, 2], dtype=DTYPE) # s, t, direction (right=1).
    cdef np.ndarray[DTYPE_t, ndim=3] incomplete = np.zeros([N+1, N+1, 2], dtype=DTYPE) # s, t, direction (right=1).     
    cdef np.ndarray[BACKTRACK_DTYPE_t, ndim=3] complete_backtrack   = -np.ones([N+1, N+1, 2], dtype=BACKTRACK_DTYPE) # s, t, direction (right=1). 
    cdef np.ndarray[BACKTRACK_DTYPE_t, ndim=3] incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=BACKTRACK_DTYPE) # s, t, direction (right=1).

    for i in range(N+1):
        incomplete[0, i, 0] = INFINITY

    cdef DTYPE_t tmp, _max

    # Loop from smaller items to larger items.
    for k in range(1, N+1):
        for s in range(N-k+1):
            t = s+k
            
            # First, create incomplete items.
            # left tree
            # incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
            # incomplete[s, t, 0] = np.max(incomplete_vals0)
            # incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            _max = -INFINITY
            for i in range(s,t):
                tmp = complete[s, i, 1] + complete[i+1, t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s]==t else 1.0)
                if tmp > _max:
                    _max = tmp
                    incomplete_backtrack[s, t, 0] = i
            # WARNING: do not pull this into the loop. causes segmentation faults
            incomplete[s, t, 0] = _max

            # right tree
            # incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t]==s else 1.0)
            # incomplete[s, t, 1] = np.max(incomplete_vals1)
            # incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)
            _max = -INFINITY
            for i in range(s,t):
                tmp = complete[s, i, 1] + complete[i+1, t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t] == s else 1.0)
                if tmp > _max:
                    _max = tmp
                    incomplete_backtrack[s, t, 1] = i
            # WARNING: do not pull this into the loop. causes segmentation faults
            incomplete[s, t, 1] = _max

            # Second, create complete items.
            # left tree
            # complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            # complete[s, t, 0] = np.max(complete_vals0)
            # complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            _max = -INFINITY            
            for i in range(s,t):
                tmp = complete[s, i, 0] + incomplete[i, t, 0]
                if tmp > _max:
                    _max = tmp
                    complete_backtrack[s, t, 0] = i
            # WARNING: do not pull this into the loop. causes segmentation faults
            complete[s, t, 0] = _max
            
            # right tree
            # complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            # complete[s, t, 1] = np.max(complete_vals1)
            # complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)
            _max = -INFINITY
            for i in range(s,t):
                tmp = incomplete[s, i+1, 1] + complete[i+1, t, 1]
                if tmp > _max:
                    _max = tmp
                    complete_backtrack[s, t, 1] = i + 1
            # WARNING: do not pull this into the loop. causes segmentation faults
            complete[s, t, 1] = _max
        
    heads = [-1 for _ in range(N+1)] #-np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    return np.array(heads, dtype=int)

@cython.boundscheck(False)
cdef void backtrack_eisner(np.ndarray[BACKTRACK_DTYPE_t, ndim=3] incomplete_backtrack, 
                           np.ndarray[BACKTRACK_DTYPE_t, ndim=3] complete_backtrack, 
                           int s, int t, int direction, int complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the 
    head of each word.
    '''
    # print(s,t)
    cdef BACKTRACK_DTYPE_t r
    if s == t:
        return
    if complete == 1:
        r = complete_backtrack[s,t,direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s,t,direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return