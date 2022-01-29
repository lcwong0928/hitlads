import glob
import json
import os

import tensorflow as tf

from src.configuration.constants import MODEL_DATA_DIRECTORY
from src.models.tadgan import TadGAN
from src.models.transformer import Encoder


class AttentionTadGAN(TadGAN):
    model_name = 'attention_tadgan'

    def __init__(self, input_shape: tuple, target_shape: tuple, num_heads: int = 5, **kwargs):
        self.num_heads = num_heads
        super(AttentionTadGAN, self).__init__(input_shape, target_shape, **kwargs)
        self.input_parameters['num_heads'] = num_heads

    def build_generator(self):
        x = tf.keras.Input(shape=self.generator_input_shape)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=self.generator_reshape_dim))
        model.add(tf.keras.layers.Reshape(target_shape=self.generator_reshape_shape))
        model.add(Encoder(num_layers=2, d_model=self.generator_reshape_shape[-1], num_heads=1,
                          dff=self.generator_reshape_shape[-1] * 4, input_vocab_size=None,
                          maximum_position_encoding=10000))

        model.add(tf.keras.layers.UpSampling1D(size=2))
        model.add(Encoder(num_layers=2, d_model=self.generator_reshape_shape[-1], num_heads=1,
                          dff=self.generator_reshape_shape[-1] * 4, input_vocab_size=None,
                          maximum_position_encoding=10000))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1)))
        model.add(tf.keras.layers.Activation(activation='tanh'))
        return tf.keras.Model(x, model(x))

    def build_encoder(self):
        x = tf.keras.Input(shape=self.encoder_input_shape)

        model = tf.keras.models.Sequential()
        model.add(Encoder(num_layers=2, d_model=self.encoder_input_shape[-1], num_heads=self.num_heads,
                          dff=self.encoder_input_shape[-1] * 4, input_vocab_size=None,
                          maximum_position_encoding=10000))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=self.dense_units))
        model.add(tf.keras.layers.Reshape(target_shape=self.encoder_reshape_shape))
        return tf.keras.Model(x, model(x))
