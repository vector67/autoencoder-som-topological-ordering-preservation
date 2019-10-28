import os

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as k
import pandas as pd

from som import create_som, plot_som
from topological import euclidean_distance2, euclidean_distance3, euclidean_distance4, get_distance_matrix, \
    triplet_ordering_measure
from tutorial import live_plotter
np.set_printoptions(precision=4)

mnist = tf.keras.datasets.mnist

num_samples_for_triplets = 5000
LATENT_SPACE_SIZE = 4
map_size = (20, 20)


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    print(features)
    new_features = tf.stack(list(features.values()), axis=1)
    return new_features, labels


def add_noise_dimensions(features, labels):
    """Pack the features into a single array."""
    print('got called')
    initializer = tf.random_uniform_initializer(0, 1)
    number_of_features = tf.shape(features)
    random_noise = initializer((number_of_features[0], 16), dtype=tf.dtypes.float32)
    print('we got called', features, random_noise)
    new_features = tf.concat([features, random_noise], 1)
    return new_features, labels


batch_size = 32

train_dataset_url = "/Users/admin/.keras/datasets/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

train, test = mnist.load_data()
images, labels = train
print(images)
print(labels)
# original_train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))


# df = pd.read_csv(train_dataset_fp, index_col=None)
# df.head()
#
# original_train_dataset = tf.data.Dataset.from_tensor_slices(dict(df))
#
# original_train_dataset = tf.data.experimental.make_csv_dataset(
#     train_dataset_fp,
#     batch_size,
#     column_names=column_names,
#     label_name=label_name,
#     num_epochs=1)
train_dataset = original_train_dataset.map(pack_features_vector)
train_dataset = train_dataset.map(add_noise_dimensions)

# print(train_dataset)
# features, labels = next(iter(train_dataset))
# print(features)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(20,)),  # input shape required
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(LATENT_SPACE_SIZE),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(20),
])

# predictions = model(features)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(x)[:, 0:4], x[:, 0:4])))
    return reconstruction_error


# l = loss(model, features, labels)
# print("Loss test: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# print(model.get_layer(index=0))
rows = []
# for i in train_dataset.unbatch().enumerate():

for i in train_dataset.unbatch().enumerate():
    row = i[1][0].numpy()
    rows.append(row)
rows = np.array(rows)
euclidean_distances = [euclidean_distance2, euclidean_distance3, euclidean_distance4]

# with a Sequential model
get_2nd_layer_output = k.function([model.layers[0].input],
                                  [model.layers[2].output])

distance_matrix_original_data = get_distance_matrix(rows)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss_results = []
train_accuracy_results = []

num_epochs = 5001
print(model.summary())

size = 10
plotting_x = np.linspace(0, 1, size + 1)[0:-1]
num_features = 4
lines = [[] for x in range(num_features)]
y_vec = [np.zeros(size) for x in range(num_features)]
labels = ['loss', 'distance_measure', 'triplet', 'triplet more samples']
first_time_multipliers = [0 for x in range(num_features)]

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # epoch_accuracy(x, model(x))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    # train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 49:
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
        # print(model.trainable_variables)

        predictions = np.zeros((rows.shape[0], LATENT_SPACE_SIZE))
        # for i in train_dataset.unbatch().enumerate():

        for i in range(len(rows)):
            row = rows[i]
            reshaped = np.array([row])
            layer_output = get_2nd_layer_output(reshaped)[0]
            # print(layer_output)

            predictions[i] = layer_output[0]

        distance_matrix = get_distance_matrix(np.array(predictions))
        number_of_values = 0
        data_size = len(predictions)
        number_of_differences = data_size * data_size - data_size

        sum_diffs = np.sum(np.abs(distance_matrix - distance_matrix_original_data))
        topological_ordering_measure_by_distance = sum_diffs / number_of_differences

        triplet_measure = triplet_ordering_measure(rows, predictions, num_samples_for_triplets)
        triplet_measure_more_samples = triplet_ordering_measure(rows, predictions, num_samples_for_triplets * 10)
        print(sum_diffs, number_of_values)
        if first_time_multipliers[0] == 0:
            first_time_multipliers[0] = 100 / epoch_loss_avg.result()
            first_time_multipliers[1] = 100 / topological_ordering_measure_by_distance
            first_time_multipliers[2] = 100 / (triplet_measure / num_samples_for_triplets)
            first_time_multipliers[3] = 100 / (triplet_measure_more_samples / num_samples_for_triplets / 10)
        y_vec[0][-1] = epoch_loss_avg.result() * first_time_multipliers[0]
        y_vec[1][-1] = topological_ordering_measure_by_distance * first_time_multipliers[1]
        y_vec[2][-1] = triplet_measure / num_samples_for_triplets * first_time_multipliers[2]
        y_vec[3][-1] = triplet_measure_more_samples / num_samples_for_triplets / 10 * first_time_multipliers[3]
        print('topological ordering,z distance_measure',
              str(round(topological_ordering_measure_by_distance * 100) / 100) + '%')
        print('topological ordering, triplet measure', triplet_measure)
        live_plotter(plotting_x, y_vec, lines, labels)
        y_vec[0] = np.append(y_vec[0][1:], 0.0)
        y_vec[1] = np.append(y_vec[1][1:], 0.0)
        y_vec[2] = np.append(y_vec[2][1:], 0.0)
        y_vec[3] = np.append(y_vec[3][1:], 0.0)

        print(sum_diffs)
        net = create_som(predictions, map_size, LATENT_SPACE_SIZE, n_iterations=1000)
        print(net)
        plot_som(net)

# Try with training vs test data set split to see if it generalizes the ordering correctly
#     (might destroy ordering on test set)
# diabetes, wines, mnist (because of images with noise) datasets. fashion-mnist

# Partially done
# create som of the dataset


# Done
#    topology of the space with a set of points
# Try with three hidden layers
# check only within rows rather than across the whole matrix for ordering
# check triplets for triangle inequality conservation
# try manual changes with maybe like 1% of points swapped or add random noise or rotations
#    to check if the measure is actually something valid. justification for
# try different sampling sizes for triplets

# For the paper:
# start with som and the idea that a topological ordering isn't necessarily preserved.
# astronomical data tends to use autoencoders without ensuring that the topological ordering is actually preserved,
# let's try to prove this (measures of stuff) and then it's a reasonable assumption that it holds,
# or it's not a reasonable assumption, look we broke it.
# assume our measures are the measure of topological ordering
# assume the person reading is slightly mentally impaired, doesn't know concepts of AI.
# remove schedule

# passive present tense
# lit review:
#   find people who have used autoencoders before soms
#   try recent stuff.
# Explain the tables of data
#   make sure that it's obvious and put it down on writing
# write hypothesizing stuff at the end of results and discussion about why this stuff might be the case
# put in github link to the code
# Explain update equation of Adam

# graph of the epochs vs loss vs quantization error vs topological errors
# scalar is lower case non bold
# vector lower case bold
# matrices capital bold
# function depends on range, scalar function - lowercase non bold. Vector domain lower case bold.
# pseudocode for the som.


# add random extra information to appendix

# cool paper about the provability of over fitting (durban Indaba x) Dr Saxe
