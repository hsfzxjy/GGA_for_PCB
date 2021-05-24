#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t

cpdef calc_L(
    np.ndarray[np.int32_t, ndim=2] x,
    np.ndarray[np.int32_t, ndim=2] A,
    int32_t[:] order,
):
    cdef int32_t n = x.shape[0]
    cdef int32_t m = x.shape[1]
    cdef int32_t p = A.shape[0]

    cdef int32_t[:, :] xv = x
    cdef int32_t[:, :] Av = A
    cdef int32_t i, j, t

    L = np.zeros((m, p, n), dtype=np.int32)
    cdef int32_t[:, :, :] Lv = L 

    for t in range(m):
        for j in range(n):
            if xv[j, t] == 0: continue
            for i in range(p):
                if Av[i, j] == 1:
                    Lv[t, i, order[j]] = 1

    return L

cpdef calc_sw(
    int32_t m,
    int32_t n,
    int32_t p,
    int32_t[:] Cs,
    np.ndarray[np.int32_t, ndim=3] L,
):
    sw = np.zeros((m, n), dtype=np.int32)
    cdef int32_t[:, :] swv = sw
    cdef int32_t i, j, loc, t, selected, instant, C, used, k, tt, cur_load, minv, maxv

    next = np.zeros((p,), dtype=np.int32)
    cdef int32_t[:] nextv = next
    slots = np.zeros((p,), dtype=np.int32)
    cdef int32_t[:] slotsv = slots

    cdef int32_t[:, :, :] Lv = L

    for t in range(m):
        C = Cs[t]

        for i in range(p):
            slotsv[i] = 0
            instant = 0
            while instant < n and Lv[t, i, instant] == 0: instant += 1
            if instant == n:
                nextv[i] = n + 1
            else:
                nextv[i] = instant

        cur_load = 0
        for instant in range(n):
            used = 0
            for i in range(p):
                if Lv[t, i, instant] == 0: continue
                used = 1
                if slotsv[i] == 0:
                    cur_load += 1
                    slotsv[i] = 1
                    swv[t, instant] += 1
                
                loc = instant + 1
                while loc < n and Lv[t, i, loc] == 0: loc += 1
                nextv[i] = loc

            if used == 0: continue
            while cur_load > C:
                maxv = 0
                for i in range(n):
                    if slotsv[i] == 0 or Lv[t, i, instant] == 1: continue
                    if maxv < nextv[i]:
                        maxv = nextv[i]
                        selected = i
                slotsv[selected] = 0
                cur_load -= 1
                swv[t, instant] += 1

            while cur_load < C:
                selected = -1
                minv = n + 1
                for i in range(n):
                    if slotsv[i] == 1: continue
                    if minv > nextv[i]:
                        minv = nextv[i]
                        selected = i
                if selected == -1:
                    break
                slotsv[selected] = 1
                cur_load += 1
                swv[t, instant] += 1

    return sw