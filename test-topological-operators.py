import math
import random
import sys
import time

import numpy as np

# from topological import get_distance_matrix
from topological import triplet_ordering_measure, get_distance_matrix, euclidean_distance2, euclidean_distance3, \
    euclidean_distance4

# row_size = 4
# my_array = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 1, 9]])
# my_array2 = np.array([[6, 3, 2, 4], [7, 3, 1, 34], [3, 8, 6, 1]])
# stacked = np.stack((my_array, my_array2), axis=-2)
# range = np.arange(3)
# num_samples = 2
# all_coordinates = np.reshape(np.array(np.meshgrid(range, range)).T, (9, 2))
# samples = np.random.choice(3, size=(num_samples, 3))
# samples_comparison = np.roll(samples, 1, axis=1)
# print(samples)
# # print(samples_comparison)
# print(my_array[samples] - my_array[samples_comparison])
# print(np.linalg.norm(my_array[samples] - my_array[samples_comparison], axis=2))
# ordering1 = np.argsort(np.linalg.norm(my_array[samples] - my_array[samples_comparison], axis=2))
# print(ordering1)
# ordering2 = np.argsort(np.linalg.norm(my_array2[samples] - my_array2[samples_comparison], axis=2))
# print(ordering2)
# print('ordering1 - ordering2')
# sum_ordering = ordering1 - ordering2
# # print(sum_ordering)
# print(np.sum(np.isclose(ordering1, ordering2)))
# print(stacked[all_coordinates])
# print(my_array[])
# print(my_array[])
# xs = np.tile(np.arange(row_size), (row_size, 1))
# ys = np.repeat(np.arange(row_size)[..., np.newaxis], row_size, axis=1)
# print(ys)
# print(np.repeat(my_array, 2, axis=0))
# print(np.repeat(my_array, 2, axis=1))
# print(np.repeat(my_array, 3, axis=0))
# print(np.repeat(my_array, (2, 1, 1, 1), axis=1))
# print(np.tile(my_array[:, np.newaxis, ...], (1, row_size, 1)))
# print(np.tile(, (1, 1, 1)))
# print(np.concatenate([my_array[np.newaxis, ...]]*3, axis=1))
# print(np.tile(my_array, (2, 2, 2)))
# print(np.repeat(my_array, (2, 2), axis=1))

euclidean_distances = [euclidean_distance2, euclidean_distance3, euclidean_distance4]

num_samples_for_triplets = 5000

mu, sigma = 50, 1
data_size = 500
rows = np.random.normal(mu, sigma, (data_size, 4))
print('completed initialization')
sys.stdout.flush()

distance_matrix_original_data = get_distance_matrix(rows)
print('completed get_distance_matrix')
sys.stdout.flush()
labels = ['loss', 'distance_measure', 'triplet', 'triplet more samples']
predictions = []
# chance_for_permutations = np.subtract(np.exp(np.linspace(1, 0, 50)), np.repeat([math.e - 1], 50))
start_exponent = 4
end_exponent = 0
chance_for_permutations = (np.exp(np.linspace(start_exponent, end_exponent, 50)) - math.exp(end_exponent)) / (
        math.exp(start_exponent) - math.exp(end_exponent)) / 4
print(chance_for_permutations)
number_of_samples = 100
results = []
outer_counter = 0

previous_time = time.time()


def debug(kwargs):
    global previous_time
    next_time = time.time()
    # print(kwargs, round((next_time - previous_time) * 100000) / 100)
    sys.stdout.flush()
    previous_time = next_time

number_of_differences = data_size*data_size - data_size

for chance_for_permutation in chance_for_permutations:
    sum_topological_ordering_measure_by_distance = 0
    sum_triplet_measure = 0
    sum_triplet_measure_more_samples = 0
    for counter in range(number_of_samples):
        new_rows = np.copy(rows)
        debug('completed copying')

        for i in range(math.floor(chance_for_permutation / 2 * data_size)):
            x1 = random.randint(0, data_size - 1)
            x2 = random.randint(0, data_size - 1)
            tmp = new_rows[x1]
            new_rows[x1] = new_rows[x2]
            new_rows[x2] = tmp
        debug('completed swapping')
        distance_matrix = get_distance_matrix(np.array(new_rows))
        debug('completed distance_matrix')
        sum_diffs = np.sum(np.abs(distance_matrix - distance_matrix_original_data))
        topological_ordering_measure_by_distance = sum_diffs / number_of_differences
        debug('completed distance measure')

        triplet_measure = triplet_ordering_measure(rows, new_rows, num_samples_for_triplets)
        debug('completed triplet measure')

        # triplet_measure_more_samples = triplet_ordering_measure(rows, new_rows, euclidean_distances[2],
        #                                                         euclidean_distances[2],
        #                                                         num_samples_for_triplets * 10)
        sum_topological_ordering_measure_by_distance += topological_ordering_measure_by_distance
        sum_triplet_measure += triplet_measure / num_samples_for_triplets
        # sum_triplet_measure_more_samples += triplet_measure_more_samples / num_samples_for_triplets / 10
        # if counter % 10 == 0:
        #     print('completed' + str(counter + 1))

    print('chance', chance_for_permutation)
    print('topological ordering, distance_measure',
          str(round(sum_topological_ordering_measure_by_distance / number_of_samples * 100) / 100) + '%')
    print('topological ordering, triplet measure', sum_triplet_measure / number_of_samples * 100)
    print(outer_counter + 1, len(chance_for_permutations))
    results.append(
        [outer_counter, chance_for_permutation, sum_topological_ordering_measure_by_distance / number_of_samples,
         sum_triplet_measure / number_of_samples * 100])
    outer_counter += 1

print(data_size)
print('num, % changed, distance_measure,triplet_measure')
for result in results:
    print(str(result[0]) + ',' + str(round(result[1] * 100) / 100) + ',' + str(
        round(result[2] * 10000) / 10000) + ',' + str(round(result[3] * 10000) / 10000))
