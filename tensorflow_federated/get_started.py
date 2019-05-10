# https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#evaluation
# Simulating a TFF model
from __future__ import absolute_import, division, print_function
import collections
from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff
from matplotlib import pyplot as plt


"""
Simple process
Step 1: Create a set of federeted data to feed to TFF model
Step 2: Create a trainable TFF model by wrapping a Keras model
Step 3: Train TFF model with federated data
"""

# Set up environment
nest = tf.contrib.framework.nest
np.random.seed(0)
tf.compat.v1.enable_v2_behavior()
print(tff.federated_computation(lambda: 'Hello, World!')())

#@test {"output": "ignore"}

# Load EMNIST dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

print(len(emnist_train.client_ids), emnist_train.output_shapes, emnist_train.output_types)

# Sample data for a client
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = iter(example_dataset).next()
print(example_element['label'].numpy())

# plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
# plt.grid('off')
# plt.show()

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
NUM_CLIENTS = 3

def preprocess(dataset):
    """
    Transform data from 28x28 images into 784-element arrays, shuffle the individual
    examples, organize them into batches, rename the features from **pixel** and
    **label** to **x** and **y** for use with Keras.

    :param dataset: tf.data.Dataset object

    :return: OrderedDict
    """
    def element_fn(element):
        return collections.OrderedDict([('x', tf.reshape(element['pixels'], [-1])),
                                        ('y', tf.reshape(element['label'], [1])),])
    return dataset.repeat(NUM_EPOCHS).map(element_fn)\
        .shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

def make_federated_data(client_data, client_ids):
    """
    Feed federated data to TFF, each element of the list holding the data of an
    individual user.

    :param client_data: Federated data of client

    :param client_ids: Client id

    :return: A list of federated data
    """
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]

def create_compiled_keras_model():
    """
    Create a Keras model
    :return: Keras model
    """
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(
        10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred))

    model.compile(loss=loss_fn,
                  optimizer=gradient_descent.SGD(learning_rate=0.02),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def model_fn():
    """
    Wrap Keras model with TFF

    :return: Trainable TFF model
    """
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = nest.map_structure(lambda  x: x.numpy(),
                                  iter(preprocessed_example_dataset).next())
# Sample 3 specific clients to get faster convergence
sample_clients = emnist_train.client_ids[0: NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)
# Initialize a trainable process with TFF
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

str(iterative_process.initialize.type_signature)
state = iterative_process.initialize()

for round_num in range(1, 50):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics = {}'.format(round_num, metrics))

# print(sample_batch)
# print(len(federated_train_data), federated_train_data[0])