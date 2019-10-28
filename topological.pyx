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
def get_distance_matrix(np.ndarray element_array):
    cdef int x, y, counter, row_size
    cdef double distance
    cdef np.ndarray[np.int64_t, ndim=2] distance_matrix
    row_size = len(element_array)
    arranged_along_x_element_array = np.tile(element_array[:, np.newaxis, ...], (1, row_size, 1))
    arranged_along_y_element_array = np.tile(element_array, (row_size, 1, 1))
    distances = np.linalg.norm(arranged_along_x_element_array - arranged_along_y_element_array, axis=2)
    new_order = np.argsort(distances, axis=1)
    distances = np.take_along_axis(distances, new_order, axis=1)
    return np.argsort(new_order, axis=1)

def euclidean_distance4(a, b):
    return abs(a[0] - b[0]) ** 0.5 + abs(a[1] - b[1]) ** 0.5 + abs(a[2] - b[2]) ** 0.5 + abs(a[3] - b[3]) ** 0.5

def euclidean_distance3(a, b):
    return abs(a[0] - b[0]) ** 0.5 + abs(a[1] - b[1]) ** 0.5 + abs(a[2] - b[2]) ** 0.5

def euclidean_distance2(a, b):
    return abs(a[0] - b[0]) ** 0.5 + abs(a[1] - b[1]) ** 0.5

def triplet_ordering_measure(points_array1, points_array2, num_samples):
    num_points = len(points_array1)
    samples = np.random.choice(num_points, size=(num_samples, 3))
    samples_comparison = np.roll(samples, 1, axis=1)
    print(samples)
    print(samples_comparison)
    print(points_array1)
    print(points_array1[samples])
    print(points_array1[samples_comparison])
    norms1 = np.linalg.norm(points_array1[samples] - points_array1[samples_comparison], axis=2)
    ordering1 = np.argsort(norms1)
    ordering2 = np.argsort(np.linalg.norm(points_array2[samples] - points_array2[samples_comparison], axis=2))
    return np.sum(np.isclose(ordering1, ordering2))
    # # print(np.take_along_axis(points_array1, samples, axis=0))
    # for i in range(num_samples):
    #     point_a = random.randint(0, num_points - 1)
    #
    #     point_b = random.randint(0, num_points - 1)
    #     while point_a == point_b:
    #         point_b = random.randint(0, num_points - 1)
    #
    #     point_c = random.randint(0, num_points - 1)
    #     while point_a == point_c or point_b == point_c:
    #         point_c = random.randint(0, num_points - 1)
    #
    #     # print(point_a, point_b, point_c)
    #     # sys.stdout.flush()
    #     distance_a1 = np.linalg.norm(points_array1[point_a] - points_array1[point_b])
    #     distance_b1 = np.linalg.norm(points_array1[point_b] - points_array1[point_c])
    #     distance_c1 = np.linalg.norm(points_array1[point_c] - points_array1[point_a])
    #
    #     distance_a2 = np.linalg.norm(points_array2[point_a] - points_array2[point_b])
    #     distance_b2 = np.linalg.norm(points_array2[point_b] - points_array2[point_c])
    #     distance_c2 = np.linalg.norm(points_array2[point_c] - points_array2[point_a])
    #
    #     ordering1 = ordering_from_distance(distance_a1, distance_b1, distance_c1)
    #     ordering2 = ordering_from_distance(distance_a2, distance_b2, distance_c2)
    #
    #     num_differences = 0
    #
    #     if not (ordering1[0] == ordering2[0]):
    #         num_differences += 1
    #
    #     if not (ordering1[1] == ordering2[1]):
    #         num_differences += 1
    #
    #     if not (ordering1[2] == ordering2[2]):
    #         num_differences += 1
    #
    #     if num_differences == 3:
    #         differences += 1
    #     elif num_differences == 1:
    #         differences += 0.5
    # return differences

def ordering_from_distance(distance_a, distance_b, distance_c):
    if distance_a > distance_b:
        if distance_b > distance_c:
            ordering = 'abc'
        else:
            if distance_a > distance_c:
                ordering = 'acb'
            else:
                ordering = 'cab'
    else:
        if distance_a > distance_c:
            ordering = 'bac'
        else:
            if distance_b > distance_c:
                ordering = 'bca'
            else:
                ordering = 'cba'
    return ordering
