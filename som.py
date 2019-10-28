from __future__ import division

import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches


#
# num_data_points = 1000
# data_dimensionality = 50
# raw_data = np.random.randint(0, 256, (num_data_points, data_dimensionality)) / 255
# network_size = (10, 10)


def create_som(data, map_size, data_dimensionality, n_iterations=100000):
    init_learning_rate = 0.01
    num_data_points = len(data)

    # initial neighbourhood radius
    init_radius = max(map_size[0], map_size[1]) / 2
    # radius decay parameter
    time_constant = n_iterations / np.log(init_radius)

    net = np.random.random((map_size[0], map_size[1], data_dimensionality))

    mesh = np.array(np.meshgrid(np.arange(net.shape[1]), np.arange(net.shape[0])))
    xy_coords_of_net = np.stack(np.array([mesh[1], mesh[0]]), axis=2)

    data_reshape_time = 0
    bmu_time = 0
    decay_time = 0
    calculate_distances_time = 0
    update_neighbours_time = 0
    sample_points = np.random.choice(num_data_points, n_iterations)
    samples = data[sample_points]

    # start_time = time.time()
    for i in range(n_iterations):
        # select a training example at random
        t1 = time.time()
        sample = samples[i]
        # sample = data[np.random.randint(0, num_data_points), :].reshape(3)

        t2 = time.time()

        bmu_idx = np.array(
            np.unravel_index(np.argmin(np.linalg.norm(net - sample.reshape(data_dimensionality), axis=2)),
                             map_size))

        t3 = time.time()
        # decay the SOM parameters
        r = init_radius * np.exp(-i / time_constant)
        current_learning_rate = init_learning_rate * np.exp(-i / n_iterations)
        t4 = time.time()
        w_dists = np.linalg.norm(bmu_idx - xy_coords_of_net, axis=2)

        t5 = time.time()
        r_constant = (2 * (r ** 2))
        influence = np.tile(np.exp(-w_dists / r_constant)[..., np.newaxis], (1, 1, data_dimensionality))
        w_dists_boolean = w_dists <= r
        influence_boolean = np.tile(w_dists_boolean[..., np.newaxis], (1, 1, data_dimensionality))

        additive = (current_learning_rate * influence * (sample - net)) * influence_boolean
        net = net + additive
        t6 = time.time()

        data_reshape_time += t2 - t1
        bmu_time += t3 - t2
        decay_time += t4 - t3
        calculate_distances_time += t5 - t4
        update_neighbours_time += t6 - t5

    return net


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
# print('data_reshape', data_reshape_time)
# print('bmu_time', bmu_time)
# print('decay_time', decay_time)
# print('calculate_distances_time', calculate_distances_time)
# print('update_neighbours_time', update_neighbours_time)
#
# print("done after:", time.time() - start_time)
# print('time taken not doing stuff', time.time() - start_time -
# data_reshape_time - bmu_time - decay_time - calculate_distances_time - update_neighbours_time)
# sys.stdout.flush()
