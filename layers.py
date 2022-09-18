# Custom L1 Distance layer


# Import dependencies

import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
    This is needed to load the siameseModel to predict
"""


class L1Dist(Layer):

    def __init__(self, **kwargs):
        # inherits the __init__ constructor from Layer class
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        # Calculates the similarity between two image embeddings

        return tf.math.abs(input_embedding - validation_embedding)


l1 = L1Dist()