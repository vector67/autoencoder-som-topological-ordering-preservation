import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
tf.Session(config=config)
print('done')


class Autoencoder(object):

    def __init__(self, input_dim, encoded_dim):
        input_layer = Input(shape=(input_dim, input_dim, 1))
        hidden_input = Input(shape=(encoded_dim,))
        hidden_layer = Dense(encoded_dim, activation='relu')(input_layer)
        output_layer = Dense(784, activation='sigmoid')(hidden_layer)

        self._autoencoder_model = Model(input_layer, output_layer)
        self._encoder_model = Model(input_layer, hidden_layer)
        tmp_decoder_layer = self._autoencoder_model.layers[-1]
        self._decoder_model = Model(hidden_input, tmp_decoder_layer(hidden_input))

        self._autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, input_train, input_test, epochs):
        self._autoencoder_model.fit(input_train,
                                    epochs=epochs,
                                    validation_data=(
                                        input_test), steps_per_epoch=3, validation_steps=1)

    def getEncodedImage(self, image):
        encoded_image = self._encoder_model.predict(image)
        return encoded_image

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._decoder_model.predict(encoded_imgs)
        return decoded_image


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=1)
    image = tf.image.resize_images(image, [500, 500])
    image /= 255
    #     image = tf.image.rgb_to_grayscale(image)
    #     print('filename', filename)
    #     print('image_grayscale', image)
    #     image_grayscale = np.reshape(image_grayscale, (500, 500))

    return image, image


# A vector of filenames.
TRAIN_BUF = 10
BATCH_SIZE = 2

image_directory = '/idia/projects/hippo/rgz/PNG-IMGS/'
train_filenames = []
test_filenames = []
with open('png_names_sorted.txt', 'r') as f:
    for i in range(500):
        train_filenames.append(image_directory + f.readline()[2:].strip())
    for i in range(200):
        test_filenames.append(image_directory + f.readline()[2:].strip())
train_filenames = tf.constant(train_filenames)
test_filenames = tf.constant(test_filenames)

# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')


# `labels[i]` is the label for the image in `filenames[i].


x_train = tf.data.Dataset.from_tensor_slices((train_filenames)).map(_parse_function).shuffle(TRAIN_BUF).batch(
    BATCH_SIZE)
x_test = tf.data.Dataset.from_tensor_slices((test_filenames)).map(_parse_function).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

x_train_iterator = x_train.make_one_shot_iterator().get_next()
x_test_iterator = x_test.make_one_shot_iterator().get_next()

# # Import data
# (x_train, _), (x_test, _) = fashion_mnist.load_data()

# # Prepare input
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Keras implementation
print('\n\n\ncreating')
autoencoder = Autoencoder(500, 32)
print('\n\n\ntraining')
x_train_batch, labels = next(iter(x_train))
x_test_batch, labels = next(iter(x_test))
print(x_train_iterator)
autoencoder.train(x_train_iterator, x_test_iterator, 1)
print('\n\n\ngetting encoded')
encoded_imgs = autoencoder.getEncodedImage(x_test)
print('\n\n\ngetting decoded')
decoded_imgs = autoencoder.getDecodedImage(encoded_imgs)

# Keras implementation results
plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    subplot = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)

    # Reconstruction
    subplot = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()
