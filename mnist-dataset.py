import datetime
import sys
import time

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as k
import matplotlib.pyplot as plt
import os

from som import create_som, calculate_quantization_error
from topological import get_distance_matrix_measure, \
    triplet_ordering_measure

np.set_printoptions(precision=4)

mnist = tf.keras.datasets.fashion_mnist

num_samples_for_triplets = 20000
num_samples_for_distance_measure = 400
batch_size = 256

map_size = (30, 30)


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    number_of_features = tf.shape(features)
    current_batch_size = number_of_features[0]
    new_features = tf.reshape(features, [current_batch_size, -1]) / 255
    return new_features, labels


def add_noise_dimensions(features, labels):
    """Pack the features into a single array."""
    number_of_features = tf.shape(features)
    current_batch_size = number_of_features[0]
    random_noise = tf.dtypes.cast(tf.random.uniform(number_of_features, minval=0, maxval=255, dtype=tf.dtypes.int64),
                                  dtype=tf.dtypes.float32) / 255
    random_noise = tf.reshape(random_noise, [-1])
    features = tf.reshape(features, [-1])
    dataset = tf.reshape(tf.stack([features, random_noise]), [current_batch_size, -1])
    return dataset, labels


train, test = mnist.load_data()
images, labels = train
# images = images[:5000]
# labels = labels[:5000]
images = images[:int(images.shape[0] / batch_size) * batch_size]
labels = labels[:int(labels.shape[0] / batch_size) * batch_size]
data_size = images.shape[1] * images.shape[2] * 2  # since we have noise
latent_space_size = int(data_size * 0.5)

original_train_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

train_dataset = original_train_dataset.map(pack_features_vector)
train_dataset = train_dataset.map(add_noise_dimensions)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(data_size * 0.75, input_shape=(data_size,)),
    tf.keras.layers.Dense(latent_space_size),
    tf.keras.layers.Dense(data_size * 0.75),
    tf.keras.layers.Dense(data_size),
])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y):
    reconstruction_error = tf.reduce_mean(
        tf.square(tf.subtract(model(x)[:, 0:latent_space_size], x[:, 0:latent_space_size])))
    return reconstruction_error


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


print('initializing rows')
sys.stdout.flush()
max_elems = images.shape[0]
dataset = train_dataset.batch(max_elems)
whole_dataset_tensors, labels_tensor = tf.data.experimental.get_single_element(dataset)
label_rows = labels_tensor.numpy()
rows = whole_dataset_tensors.numpy()
not_batched_label_rows = np.reshape(label_rows, (-1))
not_batched_rows = np.reshape(rows, (-1, rows.shape[-1]))
# num_som_iterations = max(len(not_batched_rows) * 3, map_size[0] * map_size[1] * 10)
num_som_iterations = 60000

print('done unbatching rows')
sys.stdout.flush()

get_2nd_layer_output = k.function([model.layers[0].input],
                                  [model.layers[1].output])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss_results = []
train_accuracy_results = []

num_epochs = 50

dataset_name = 'fashion-mnist'
final_data = [['epoch_number',
               'dataset',
               'loss',
               'topological_distance_measure',
               'topological_triplet_measure',
               'quantization_error',
               'som_file']]
directory = 'data/' + dataset_name + '-' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
with open(directory + 'metadata.txt', 'w') as f:
    f.write('dataset: ' + str(dataset_name) + '\n')
    f.write('num_samples_for_triplets: ' + str(num_samples_for_triplets) + '\n')
    f.write('num_samples_for_distance_measure: ' + str(num_samples_for_distance_measure) + '\n')
    f.write('batch_size: ' + str(batch_size) + '\n')
    f.write('map_size: ' + str(map_size) + '\n')
    f.write('num_epochs: ' + str(num_epochs) + '\n')
    f.write('optimizer: Adam\n')
    f.write('num_elements: ' + str(len(not_batched_rows)) + '\n')
    f.write('latent_space_size: ' + str(latent_space_size) + '\n')
    f.write('num_som_iterations: ' + str(num_som_iterations) + '\n')

with open(directory + 'autoencoder-setup.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print('done with init, ')
last_quantization_error = 0
for epoch in range(num_epochs):
    last_time = time.time()
    epoch_loss_avg = tf.keras.metrics.Mean()

    print('starting epoch', epoch)
    # Training loop - using batches of batch_size
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())

    print("Epoch {:03d} {:.3f}".format(epoch, time.time() - last_time))
    sys.stdout.flush()
    loss_value_at_epoch = epoch_loss_avg.result()
    print("Loss: {:.3f}".format(loss_value_at_epoch))

    predictions = np.zeros((rows.shape[0], rows.shape[1], latent_space_size))

    for i in range(len(rows)):
        layer_output = get_2nd_layer_output(rows[i])[0]
        predictions[i] = layer_output[0]

    predictions = np.reshape(predictions, (-1, predictions.shape[-1]))
    print(predictions.shape)
    print('calculating distance_matrix_measure')
    topological_ordering_measure_by_distance = get_distance_matrix_measure(not_batched_rows, predictions,
                                                                           num_samples_for_distance_measure)
    print('done calculating distance_matrix_measure')

    print('calculating triplet ordering measure')
    triplet_measure = triplet_ordering_measure(not_batched_rows, predictions, num_samples_for_triplets)
    print('done calculating triplet ordering measure')
    print('topological ordering, distance_measure',
          str(round(topological_ordering_measure_by_distance * 100) / 100) + '%')
    print('topological ordering, triplet measure', triplet_measure)
    sys.stdout.flush()
    if epoch % 10 == 0:
        print('creating som')
        sys.stdout.flush()
        net = create_som(predictions, map_size, latent_space_size, not_batched_label_rows,
                         n_iterations=num_som_iterations)
        print('created som')
        sys.stdout.flush()

        som_file = directory + 'som' + str(epoch) + '.txt'
        with open(som_file, 'w') as f:
            net.tofile(f, ',')
        last_quantization_error, label_predictions = calculate_quantization_error(net, predictions,
                                                                                  not_batched_label_rows)

        plt.figure(figsize=(8, 8))
        for x in range(label_predictions.shape[0]):
            for y in range(label_predictions.shape[1]):
                w = label_predictions[x][y]
                if w >= 0:
                    plt.text(x + 0.5, y + .5, str(int(w)),
                             color=plt.cm.rainbow(w / 10.), fontdict={'weight': 'bold', 'size': 11})
        plt.axis([0, label_predictions.shape[0], 0, label_predictions.shape[1]])
        plt.savefig(directory + 'som' + str(epoch) + '.png')
        plt.show()
    final_data.append([epoch,
                       dataset_name,
                       "Loss: {:.3f}".format(loss_value_at_epoch),
                       topological_ordering_measure_by_distance,
                       triplet_measure,
                       last_quantization_error])
    with open(directory + 'final_data.txt', 'w') as f:
        for values in final_data:
            f.write(','.join(map(str, values)))
            f.write('\n')
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
