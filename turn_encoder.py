"""
@author sourabhxiii
"""

import numpy as np
import tensorflow as tf

BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 100


class Encoder(tf.keras.layers.Layer):
    def __init__(self
                , embedding_matrix
                , latent_dim=128
                , name='encoder'
                , **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        self.bigru = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(128, return_sequences=True
                        , dropout=0.2))
        self.gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
        self.dropout = tf.keras.layers.Dropout(0.4)
    

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.bigru(x)
        x = self.dropout(x)
        o, h = self.gru(x)
        return o, h


class ConditionalEncoder(tf.keras.layers.Layer):
    def __init__(self
                , embedding_matrix
                , latent_dim=128
                , name='encoder'
                , **kwargs):
        super(ConditionalEncoder, self).__init__(name=name, **kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        self.gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
        self.dropout = tf.keras.layers.Dropout(0.4)
    

    def call(self, inputs, latent_inputs):
        x = self.embedding_layer(inputs)
        o, h = self.gru(x, initial_state=[latent_inputs])
        return o, h

class ExitLayer(tf.keras.layers.Layer):
    def __init__(self
                , name='ExitLayer'
                , **kwargs):
        super(ExitLayer, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(128)
        self.prelu = tf.keras.layers.PReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.4)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.bn(x)
        return x