import glob
import json
import os
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from src.configuration.constants import MODELS_DIRECTORY
from src.models.transformer import Encoder
from tqdm import tqdm


def _wasserstein_loss(y_true, y_hat):
    return K.mean(y_true * y_hat)


def _gradient_penalty_loss(y_true, y_hat, averaged_samples):
    gradients = K.gradients(y_hat, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = K.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class TadGAN:
    model_name = 'tadgan'

    def __init__(self, input_shape: tuple, target_shape: tuple, latent_dim: int = 20, lstm_units: int = 100,
                 dense_units: int = 20, learning_rate: float = 0.0005, epochs: int = 70, batch_size: int = 64,
                 iterations_critic: int = 5, **kwargs):

        # Input Parameters
        self.input_parameters = locals()
        del self.input_parameters['self']
        del self.input_parameters['kwargs']

        self.input_shape = input_shape
        self.target_shape = target_shape
        self.length = input_shape[0]

        self.latent_dim = latent_dim
        self.latent_shape = (self.latent_dim, 1)

        self.generator_reshape_dim = (self.length // 2) * target_shape[-1]
        self.generator_reshape_shape = (self.length // 2, target_shape[-1])
        self.encoder_reshape_shape = self.latent_shape

        self.encoder_input_shape = input_shape
        self.generator_input_shape = self.latent_shape
        self.critic_x_input_shape = target_shape
        self.critic_z_input_shape = self.latent_shape

        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations_critic = iterations_critic
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Tensorflow Models
        self.encoder = self.build_encoder()
        self.generator = self.build_generator()
        self.critic_z = self.build_critic_z()
        self.critic_x = self.build_critic_x()
        self.critic_x_model, self.critic_z_model, self.encoder_generator_model = self.build_tadgan()

        # Epoch Losses
        self.epoch_loss = kwargs.get('epoch_loss', [])

    def fit(self, x_train, y_train, print_logs=False):

        fake = np.ones((self.batch_size, 1))
        valid = -np.ones((self.batch_size, 1))
        delta = np.ones((self.batch_size, 1))

        indices = np.arange(x_train.shape[0])

        for epoch in tqdm(range(len(self.epoch_loss) + 1, self.epochs + 1)):
            np.random.shuffle(indices)
            X_ = x_train[indices]
            y_ = y_train[indices]

            epoch_g_loss = []
            epoch_cx_loss = []
            epoch_cz_loss = []

            minibatches_size = self.batch_size * self.iterations_critic
            num_minibatches = int(X_.shape[0] // minibatches_size)

            for i in range(num_minibatches):
                minibatch = X_[i * minibatches_size: (i + 1) * minibatches_size]
                y_minibatch = y_[i * minibatches_size: (i + 1) * minibatches_size]

                for j in range(self.iterations_critic):
                    x = minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    y = y_minibatch[j * self.batch_size: (j + 1) * self.batch_size]
                    z = np.random.normal(size=(self.batch_size, self.latent_dim, 1))
                    epoch_cx_loss.append(
                        self.critic_x_model.train_on_batch([y, z], [valid, fake, delta]))
                    epoch_cz_loss.append(
                        self.critic_z_model.train_on_batch([x, z], [valid, fake, delta]))

                epoch_g_loss.append(
                    self.encoder_generator_model.train_on_batch([x, z], [valid, valid, y]))

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            g_loss = np.mean(np.array(epoch_g_loss), axis=0)

            self.epoch_loss.append([cx_loss, cz_loss, g_loss])

            if print_logs:
                print(f'Epoch: {epoch}/{self.epochs}, [Dx loss: {cx_loss}] [Dz loss: {cz_loss}] [G loss: {g_loss}]')

    def predict(self, x_test, y_test) -> tuple:
        z_ = self.encoder.predict(x_test)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(y_test)
        return y_hat, critic

    def fit_predict(self, x_train, y_train) -> tuple:
        self.fit(x_train, y_train)
        return self.predict(x_train, y_train)

    def build_encoder(self):
        x = tf.keras.Input(shape=self.encoder_input_shape)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.lstm_units, return_sequences=True)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=self.dense_units))
        model.add(tf.keras.layers.Reshape(target_shape=self.encoder_reshape_shape))
        return tf.keras.Model(x, model(x))

    def build_critic_z(self):
        x = tf.keras.Input(shape=self.critic_z_input_shape)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())

        for _ in range(2):
            model.add(tf.keras.layers.Dense(units=self.dense_units))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            model.add(tf.keras.layers.Dropout(rate=0.2))

        model.add(tf.keras.layers.Dense(units=1))
        return tf.keras.Model(x, model(x))

    def build_generator(self):
        x = tf.keras.Input(shape=self.generator_input_shape)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=self.generator_reshape_dim))
        model.add(tf.keras.layers.Reshape(target_shape=self.generator_reshape_shape))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat'))

        model.add(tf.keras.layers.UpSampling1D(size=2))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1)))
        model.add(tf.keras.layers.Activation(activation='tanh'))
        return tf.keras.Model(x, model(x))

    def build_critic_x(self):
        x = tf.keras.Input(shape=self.critic_x_input_shape)
        model = tf.keras.models.Sequential()

        for _ in range(4):
            model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=1))
        return tf.keras.Model(x, model(x))

    def build_tadgan(self) -> tuple:
        self.generator.trainable = False
        self.encoder.trainable = False

        x = tf.keras.Input(shape=self.input_shape)
        y = tf.keras.Input(shape=self.target_shape)
        z = tf.keras.Input(shape=(self.latent_dim, 1))

        x_ = self.generator(z)
        z_ = self.encoder(x)
        fake_x = self.critic_x(x_)  # Fake
        valid_x = self.critic_x(y)  # Truth

        # Critic X Model
        interpolated_x = RandomWeightedAverage(self.batch_size)([y, x_])
        validity_interpolated_x = self.critic_x(interpolated_x)
        partial_gp_loss_x = partial(_gradient_penalty_loss, averaged_samples=interpolated_x)
        partial_gp_loss_x.__name__ = 'gradient_penalty'
        self.critic_x_model = tf.keras.Model(inputs=[y, z], outputs=[valid_x, fake_x, validity_interpolated_x])
        self.critic_x_model.compile(loss=[_wasserstein_loss, _wasserstein_loss, partial_gp_loss_x],
                                    optimizer=self.optimizer, loss_weights=[1, 1, 10])

        # Critic Z Model
        fake_z = self.critic_z(z_)
        valid_z = self.critic_z(z)
        interpolated_z = RandomWeightedAverage()([z, z_])
        validity_interpolated_z = self.critic_z(interpolated_z)
        partial_gp_loss_z = partial(_gradient_penalty_loss, averaged_samples=interpolated_z)
        partial_gp_loss_z.__name__ = 'gradient_penalty'
        self.critic_z_model = tf.keras.Model(inputs=[x, z], outputs=[valid_z, fake_z, validity_interpolated_z])
        self.critic_z_model.compile(loss=[_wasserstein_loss, _wasserstein_loss, partial_gp_loss_z],
                                    optimizer=self.optimizer, loss_weights=[1, 1, 10])

        # Encoder Generator Model
        self.critic_x.trainable = False
        self.critic_z.trainable = False
        self.generator.trainable = True
        self.encoder.trainable = True

        z_gen = tf.keras.Input(shape=(self.latent_dim, 1))
        x_gen_ = self.generator(z_gen)
        x_gen = tf.keras.Input(shape=self.input_shape)
        z_gen_ = self.encoder(x_gen)
        x_gen_rec = self.generator(z_gen_)
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        self.encoder_generator_model = tf.keras.Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        self.encoder_generator_model.compile(loss=[_wasserstein_loss, _wasserstein_loss, 'mse'],
                                             optimizer=self.optimizer, loss_weights=[1, 1, 10])

        return self.critic_x_model, self.critic_z_model, self.encoder_generator_model

    def save(self, source: str, dataset: str, signal: str):
        output_directory = os.path.join(MODELS_DIRECTORY, source, dataset, signal, self.model_name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.encoder.save(os.path.join(output_directory, 'encoder.model'))
        self.critic_z.save(os.path.join(output_directory, 'critic_z.model'))
        self.generator.save(os.path.join(output_directory, 'generator.model'))
        self.critic_x.save(os.path.join(output_directory, 'critic_x.model'))

        self.input_parameters['epoch_loss'] = np.array(self.epoch_loss).tolist()

        with open(os.path.join(output_directory, 'input_parameters.json'), 'w') as f:
            json.dump(self.input_parameters, f)

    @classmethod
    def load(cls, source: str, dataset: str, signal: str):
        model_directory = os.path.join(MODELS_DIRECTORY, source, dataset, signal, cls.model_name)
        with open(os.path.join(model_directory, 'input_parameters.json'), 'rb') as f:
            input_parameters = json.load(f)
        model = cls(**input_parameters)
        for model_path in glob.glob(os.path.join(model_directory, '**.model')):
            model_name = os.path.basename(model_path).split('.')[0]
            model.__setattr__(model_name, tf.keras.models.load_model(model_path, custom_objects={'Encoder': Encoder}))
        model.build_tadgan()
        return model
