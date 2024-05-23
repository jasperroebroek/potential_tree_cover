# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from typing import Dict, List

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

ctypedef np.uint8_t uint8

cdef struct IterParams:
    size_t[2] stop
    size_t[2] step
    size_t[2] iter


cdef IterParams * _define_iter_params(size_t[2] shape,
                                      size_t[2] window_size) nogil:
    cdef IterParams * ip = <IterParams *> malloc(sizeof(IterParams))

    ip.stop[0] = shape[0]
    ip.stop[1] = shape[1]
    ip.step[0] = window_size[0]
    ip.step[1] = window_size[1]

    ip.iter[0] = ip.stop[0] / ip.step[0]
    ip.iter[1] = ip.stop[1] / ip.step[1]

    return ip


cdef _define_sampling(float[:, ::1] target,   # target
                      int[:, ::1] lcc,  # landcover class
                      size_t window_size,
                      uint8[:, ::1] sampling_mask,
                      int[::1] classes,  # landcover classes
                      int target_selected): # switch: 0 -> 0; -1 = pick target
    """Returns: tuple of (sampling mask, sampled treecover, weights)"""

    cdef:
        size_t p, q, i, j, x, y   # iteration counters
        size_t c  # iteration over classes

        size_t num_classes
        float tc_max
        size_t shape[2]
        size_t ws[2]

    shape[0] = target.shape[0]
    shape[1] = target.shape[1]
    ws[0] = window_size
    ws[1] = window_size
    num_classes = len(classes)

    ip = _define_iter_params(shape, ws)

    for y in range(ip.iter[0]):
        for x in range(ip.iter[1]):
            i = y * ip.step[0]
            j = x * ip.step[1]

            if target_selected == 0:
                for p in range(window_size):
                    for q in range(window_size):
                        for c in range(num_classes):
                            if lcc[i + p, j + q] == classes[c]:
                                sampling_mask[i + p, j + q] = True
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

            elif target_selected == -1:
                tc_max = -1

                for p in range(window_size):
                    for q in range(window_size):
                        for c in range(num_classes):
                            if lcc[i + p, j + q] == classes[c] and target[i + p, j + q] > tc_max:
                                tc_max = target[i + p, j + q]
                                break

                for p in range(window_size):
                    for q in range(window_size):
                        for c in range(num_classes):
                            if lcc[i + p, j + q] == classes[c] and target[i + p, j + q] == tc_max:
                                sampling_mask[i + p, j + q] = True
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

    free(ip)


cpdef define_sampling_mask(float[:, ::1] target,
                           int[:, ::1] lcc,
                           int window_size,
                           class_divisions: Dict[str, List[float]],
                           class_tc: Dict[str, int]):

    cdef:
        uint8[:, ::1] r = np.zeros_like(target, dtype=bool) # sampling mask
        int[::1] classes

    for current_class in class_divisions:
        classes = np.asarray(class_divisions[current_class], np.int32)
        _define_sampling(target=target, lcc=lcc, window_size=window_size, sampling_mask=r, classes=classes,
                         target_selected=class_tc[current_class])

    return r.base
