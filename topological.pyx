import random
import sys
import time

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t
dtype_tuple = [('distance', float), ('x', int), ('y', int)]

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_distance_matrix_measure(np.ndarray element_array1, np.ndarray element_array2, int num_samples):
    cdef int x, y, counter, row_size
    cdef double distance
    cdef np.ndarray[np.int64_t, ndim=2] distance_matrix

    samples = np.random.choice(element_array1.shape[0], size=num_samples)
    sample_array1 = element_array1[samples]
    sample_array2 = element_array2[samples]
    final_order1 = get_ordering_from_samples(num_samples, sample_array1)
    final_order2 = get_ordering_from_samples(num_samples, sample_array2)

    number_of_differences = num_samples * num_samples - num_samples

    sum_diffs = np.sum(np.abs(final_order1 - final_order2))
    return sum_diffs / number_of_differences

def get_ordering_from_samples(num_samples, sample_array):
    arranged_along_x_element_array = np.tile(sample_array[:, np.newaxis, ...], (1, num_samples, 1))
    arranged_along_y_element_array = np.tile(sample_array, (num_samples, 1, 1))
    distances = np.linalg.norm(arranged_along_x_element_array - arranged_along_y_element_array, axis=2)
    new_order = np.argsort(distances, axis=1)
    return np.argsort(new_order, axis=1)

def triplet_ordering_measure(points_array1, points_array2, num_samples):
    num_points = len(points_array1)
    samples = np.random.choice(num_points, size=(num_samples, 3))
    samples_comparison = np.roll(samples, 1, axis=1)
    # print('samples', samples)
    # print('samples_c', samples_comparison)
    ordering1 = np.argsort(np.linalg.norm(points_array1[samples] - points_array1[samples_comparison], axis=2))
    # print('ordering', ordering1)
    ordering2 = np.argsort(np.linalg.norm(points_array2[samples] - points_array2[samples_comparison], axis=2))
    # print('ordering', ordering1)
    return num_samples * 3 - np.sum(np.isclose(ordering1, ordering2))