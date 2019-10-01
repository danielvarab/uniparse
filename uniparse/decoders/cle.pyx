# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

cimport numpy as np
cimport cython

import numpy as np
from numpy.math cimport INFINITY

DTYPE = np.float64
INDEX_DTYPE = np.int32
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INDEX_DTYPE_t

def parse_nonproj(DTYPE_t[:,:] scores):
    """
    Parse using Chu-Liu-Edmonds algorithm.
    """
    cdef int nr, nc
    nr = scores.shape[0]
    nc = scores.shape[1]
    # nr, nc = np.shape(scores)
    
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    cdef int nw = nr - 1

    reps = []
    cdef INDEX_DTYPE_t[:] curr_nodes = np.ones(nw+1, dtype=INDEX_DTYPE)
    cdef INDEX_DTYPE_t[:,:] old_I = -np.ones((nw+1, nw+1), dtype=INDEX_DTYPE)
    cdef INDEX_DTYPE_t[:,:] old_O = -np.ones((nw+1, nw+1), dtype=INDEX_DTYPE)

    cdef int i, j
    for i in range(0, nw+1):
        reps.append({i: 0})
        for j in range(0, nw+1):
            old_I[i, j] = i
            old_O[i, j] = j
            if i == j or j == 0:
                continue

    scores_copy = scores.copy()
    final_edges = chu_liu_edmonds(scores_copy, curr_nodes, old_I, old_O, {}, reps)
    cdef np.ndarray[INDEX_DTYPE_t, ndim=1] heads = np.zeros(nw+1, dtype=INDEX_DTYPE)
    heads[0] = -1
    for key in final_edges.keys():
        ch = key
        pr = final_edges[key]
        heads[int(ch)] = pr

    return heads

cdef dict chu_liu_edmonds(DTYPE_t[:,:] scores,
                    INDEX_DTYPE_t[:] curr_nodes,
                    INDEX_DTYPE_t[:,:] old_I,
                    INDEX_DTYPE_t[:,:] old_O,
                    dict final_edges, list reps):
    """
    Chu-Liu-Edmonds algorithm
    """

    # need to construct for each node list of nodes they represent (here only!)
    cdef int nw = curr_nodes.shape[0] - 1
    # nw = np.size(curr_nodes) - 1

    # create best graph
    cdef INDEX_DTYPE_t[:] par = -np.ones(nw+1, dtype=INDEX_DTYPE)

    cdef DTYPE_t max_score

    cdef int m, h, l, lorg, li, l1, rep
    for m in range(1, nw+1):
        # only interested in current nodes
        if 0 == curr_nodes[m]:
            continue
        max_score = scores[0, m]
        par[m] = 0
        for h in range(nw+1):
            if m == h:
                continue
            if 0 == curr_nodes[h]:
                continue
            if scores[h, m] > max_score:
                max_score = scores[h, m]
                par[m] = h

    # find a cycle
    cdef dict cycle
    cdef list cycles = []
    cdef int n_cycles
    cdef int[:] added = np.zeros(nw+1, dtype=INDEX_DTYPE)
    for m in range(0, nw+1):
        n_cycles = len(cycles)
        if n_cycles > 0:
            break
        if added[m] or 0 == curr_nodes[m]:
            continue
        added[m] = 1
        cycle = {m: 0}
        l = m
        while True:
            if par[l] == -1:
                added[l] = 1
                break
            if par[l] in cycle:
                cycle = {}
                lorg = par[l]
                cycle[lorg] = par[lorg]
                added[lorg] = 1
                l1 = par[lorg]
                while l1 != lorg:
                    cycle[l1] = par[l1]
                    added[l1] = True
                    l1 = par[l1]
                cycles.append(cycle)
                break
            cycle[l] = 0
            l = par[l]
            if added[l] and (l not in cycle):
                break
            added[l] = 1

    # get all edges and return them
    n_cycles = len(cycles)
    cdef DTYPE_t pr, ch
    cdef int _a, _b
    if n_cycles == 0:
        for m in range(0, nw+1):
            if 0 == curr_nodes[m]:
                continue
            if par[m] != -1:
                _a = par[m]
                pr = old_I[_a, m]
                ch = old_O[_a, m]
                final_edges[ch] = pr
            else:
                final_edges[0] = -1
        return final_edges

    max_cyc = 0
    wh_cyc = 0
    n_cycles = len(cycles)
    for cycle in cycles:
        if np.size(cycle.keys()) > max_cyc:
            max_cyc = np.size(cycle.keys())
            wh_cyc = cycle

    cycle = wh_cyc
    cdef list cyc_nodes = list(cycle.keys())
    rep = cyc_nodes[0]

    cdef DTYPE_t cyc_weight = 0.0
    cdef int node
    cdef int n_nodes = len(cyc_nodes)
    cdef int k
    for k in range(k):
        node = cyc_nodes[node]
        cyc_weight += scores[par[node], node]

    cdef INDEX_DTYPE_t i, j1, wh1, wh
    cdef DTYPE_t max1, max2, scr
    for i in range(0, nw+1):
        if 0 == curr_nodes[i] or (i in cycle):
            continue

        max1 = -INFINITY
        wh1 = -1
        max2 = -INFINITY
        wh2 = -1

        #for j1 in cyc_nodes:
        for k in range(n_nodes):
            j1 = cyc_nodes[k]
            if scores[j1, i] > max1:
                max1 = scores[j1, i]
                wh1 = j1

            # cycle weight + new edge - removal of old
            scr = cyc_weight + scores[i, j1] - scores[par[j1], j1]
            if scr > max2:
                max2 = scr
                wh2 = j1

        scores[rep, i] = max1
        old_I[rep, i] = old_I[wh1, i]
        old_O[rep, i] = old_O[wh1, i]
        scores[i, rep] = max2
        old_O[i, rep] = old_O[i, wh2]
        old_I[i, rep] = old_I[i, wh2]

    cdef list rep_cons = []
    cdef dict rep_con
    for i in range(0, np.size(cyc_nodes)):
        rep_con = {}
        keys = sorted(reps[int(cyc_nodes[i])].keys())
        for key in keys:
            rep_con[key] = 0
        rep_cons.append(rep_con)

    # don't consider not representative nodes
    # these nodes have been folded
    for node in cyc_nodes[1:]:
        curr_nodes[node] = 0
        for key in reps[int(node)]:
            reps[int(rep)][key] = 0

    chu_liu_edmonds(scores, curr_nodes, old_I, old_O, final_edges, reps)

    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one.
    wh = -1
    found = False
    cdef int n_rep_cons = len(rep_cons)
    for i in range(n_rep_cons):
        if found:
            break
        for key in rep_cons[i]:
            if found:
                break
            if key in final_edges:
                wh = cyc_nodes[i]
                found = True

    l = par[wh]
    while l != wh:
        ch = old_O[par[l], l]
        pr = old_I[par[l], l]
        final_edges[ch] = pr
        l = par[l]

    return final_edges