from __future__ import division

import datetime
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches


#
# num_data_points = 1000
# data_dimensionality = 50
# raw_data = np.random.randint(0, 256, (num_data_points, data_dimensionality)) / 255
# network_size = (10, 10)
from minisom import MiniSom


def calculate_quantization_error(net, data, labels):
    net_map_size = (net.shape[0], net.shape[1])
    quantization_error = 0
    label_bins = [[[] for y in range(net.shape[1])] for x in range(net.shape[0])]
    # print(label_bins.shape)
    for i in range(len(data)):
        element = data[i]
        bmu_idx = np.array(
            np.unravel_index(np.argmin(np.linalg.norm(net - element, axis=2)),
                             net_map_size))
        # print(bmu_idx)
        quantization_error += np.linalg.norm(net[bmu_idx] - element)
        label_bins[bmu_idx[0]][bmu_idx[1]].append(labels[i])
    label_predictions = np.zeros(net_map_size) - 1
    for x in range(net_map_size[0]):
        predicted_label_row = label_bins[x]
        for y in range(net_map_size[1]):
            if len(predicted_label_row[y]) > 0:
                nparray = np.array(predicted_label_row[y], dtype=np.int)
                label_predictions[x][y] = np.argmax(np.bincount(nparray))
    return quantization_error, label_predictions


def create_som(data, map_size, data_dimensionality, labels, n_iterations=100000):
    som = MiniSom(map_size[0], map_size[1], data_dimensionality, sigma=max(map_size[0], map_size[1]) / 2,
                  learning_rate=0.02, neighborhood_function='triangle')
    som.pca_weights_init(data)
    print("Training...")
    som.train_random(data, n_iterations)  # random training
    # class_assignments = som.labels_map(data, labels)
    # print(class_assignments)
    print(som.get_weights())
    return som.get_weights()
    # init_learning_rate = 0.3
    # num_data_points = len(data)
    #
    # # initial neighbourhood radius
    # init_radius = max(map_size[0], map_size[1]) / 2
    #
    # net = np.random.random((map_size[0], map_size[1], data_dimensionality))
    #
    # mesh = np.array(np.meshgrid(np.arange(net.shape[1]), np.arange(net.shape[0])))
    # xy_coords_of_net = np.stack(np.array([mesh[1], mesh[0]]), axis=2)
    #
    # data_reshape_time = 0
    # bmu_time = 0
    # decay_time = 0
    # calculate_distances_time = 0
    # update_neighbours = 0
    #
    # sample_points = np.random.choice(num_data_points, n_iterations)
    # samples = data[sample_points]
    #
    # # start_time = time.time()
    # moving_average_change = [10 for x in range(20)]
    # for i in range(n_iterations):
    #     # select a training example at random
    #     t1 = time.time()
    #     sample = samples[i]
    #     # sample = data[np.random.randint(0, num_data_points), :].reshape(3)
    #
    #     t2 = time.time()
    #
    #     bmu_idx = np.array(
    #         np.unravel_index(np.argmin(np.linalg.norm(net - sample, axis=2)),
    #                          map_size))
    #
    #     t3 = time.time()
    #     # decay the SOM parameters
    #     r = init_radius * np.exp(-i / n_iterations)
    #     current_learning_rate = init_learning_rate * np.exp(-i / n_iterations)
    #     t4 = time.time()
    #     w_dists = np.linalg.norm(bmu_idx - xy_coords_of_net, axis=2)
    #
    #     t5 = time.time()
    #     r_constant = (2 * (r ** 2))
    #     influence = np.exp(-w_dists / r_constant)
    #     # influence = np.tile(np.exp(-w_dists / r_constant)[..., np.newaxis], (1, 1, data_dimensionality))
    #     w_dists_boolean = w_dists <= r
    #     # influence_boolean = np.tile(w_dists_boolean[..., np.newaxis], (1, 1, data_dimensionality))
    #     influence_boolean = w_dists_boolean
    #     # print(influence)
    #     # print(influence_boolean)
    #     # sys.exit()
    #     additive = ((current_learning_rate * (sample - net)).swapaxes(0, -1) * influence * influence_boolean).swapaxes(
    #         0, -1)
    #
    #     net = net + additive
    #     t6 = time.time()
    #
    #     data_reshape_time += t2 - t1
    #     bmu_time += t3 - t2
    #     decay_time += t4 - t3
    #     calculate_distances_time += t5 - t4
    #     update_neighbours += t6 - t5
    #     if i % 1000 == 0:
    #         total_change_per_item = np.sum(np.abs(additive)) / np.sum(w_dists_boolean)
    #         moving_average_change = moving_average_change[1:]
    #         moving_average_change.append(total_change_per_item)
    #         if sum(moving_average_change) / 20 < 0.1:
    #             break
    #         print(i, '/', n_iterations, ':', total_change_per_item, sum(moving_average_change) / 20)
    #         print(r, current_learning_rate)
    #         sys.stdout.flush()
    #         last_quantization_error, label_predictions = calculate_quantization_error(net, data, labels)
    #         print(last_quantization_error)
    #
    #         plt.figure(figsize=(8, 8))
    #         for x in range(label_predictions.shape[0]):
    #             for y in range(label_predictions.shape[1]):
    #                 w = label_predictions[x][y]
    #                 if w >= 0:
    #                     plt.text(x + 0.5, y + .5, str(int(w)),
    #                              color=plt.cm.rainbow(w / 10.), fontdict={'weight': 'bold', 'size': 11})
    #         plt.axis([0, label_predictions.shape[0], 0, label_predictions.shape[1]])
    #         plt.savefig('tmp/' + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) +'.png')
    #
    # print('data_reshape', data_reshape_time)
    # print('bmu_time', bmu_time)
    # print('decay_time', decay_time)
    # print('calculate_distances_time', calculate_distances_time)
    # print('update_neighbours', update_neighbours)
    # return net


def plot_som(net):
    fig = plt.figure()

    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, net.shape[0] + 1))
    ax.set_ylim((0, net.shape[1] + 1))
    ax.set_title('Self-Organising Map ')

    # plot
    for x in range(1, net.shape[0] + 1):
        for y in range(1, net.shape[1] + 1):
            ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                           facecolor=net[x - 1, y - 1, 0:3],
                                           edgecolor='none'))
    plt.show()
# create_som(raw_data, network_size, data_dimensionality)

#
#
# print("done after:", time.time() - start_time)
# print('time taken not doing stuff', time.time() - start_time -
# data_reshape_time - bmu_time - decay_time - calculate_distances_time - update_neighbours_time)
# sys.stdout.flush()
