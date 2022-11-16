from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class VanillaVAE(tf.keras.Model):
    def __init__(self, image_shape, latent_dim):
        super(VanillaVAE, self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.shape_before_flattening = None
        self.encoder = None
        self.decoder = None

        self.train_total_loss_tracker = tf.keras.metrics.Mean(
            name="train_total_loss")
        self.train_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="train_reconstruction_loss"
        )
        self.train_kl_loss_tracker = tf.keras.metrics.Mean(
            name="train_kl_loss")

        self.test_total_loss_tracker = tf.keras.metrics.Mean(
            name="total_loss")
        self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.test_kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        self.build_vae()

    def encoder_block(self, inputs, filters, kernel_size, strides, padding, activation, layer, use_bn=False):
        x = layers.Conv2D(filters, kernel_size, strides,
                          padding, activation=activation, name='Conv2D_'+str(layer))(inputs)
        if use_bn:
            x = layers.BatchNormalization(name='BN_'+str(layer))(x)
        return x

    def decoder_block(self, inputs, filters, kernel_size, strides, padding, activation, layer, use_bn=False):
        x = layers.Conv2DTranspose(
            filters, kernel_size, strides, padding, activation=activation, name='Conv2DTranspose_'+str(layer))(inputs)
        if use_bn:
            x = layers.BatchNormalization(name='BN_'+str(layer))(x)
        return x

    def reparameterize(self, coding):
        mu, log_var = coding
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var * .5) * epsilon

    def build_encoder(self):
        inputs = layers.Input(shape=self.image_shape, name='Encoder_Input')
        x = self.encoder_block(inputs=inputs, filters=32, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=1, use_bn=True)
        x = self.encoder_block(inputs=x, filters=64, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=2, use_bn=True)
        x = self.encoder_block(inputs=x, filters=128, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=3, use_bn=True)
        x = self.encoder_block(inputs=x, filters=256, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=4, use_bn=True)
        x = self.encoder_block(inputs=x, filters=512, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=5, use_bn=True)
        self.shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
        x = layers.Flatten(name='Flatten')(x)
        x = layers.Dense(512, activation='relu', name='Dense_Layer_6')(x)
        mu = layers.Dense(self.latent_dim, name='Dense_MU')(x)
        log_var = layers.Dense(self.latent_dim, name='Dense_Log_Var',
                               kernel_initializer=tf.keras.initializers.RandomUniform(
                                   minval=-0.08, maxval=0.08, seed=None)
                               )(x)
        z = layers.Lambda(self.reparameterize, output_shape=(
            self.latent_dim,), name='Code')([mu, log_var])

        self.encoder = tf.keras.Model(inputs, [mu, log_var, z], name='encoder')

    def build_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,), name='Decoder_Input')
        x = layers.Dense(512, activation='relu', name='Dense_Layer_7')(inputs)
        x = layers.Dense(np.prod(self.shape_before_flattening),
                         activation='relu', name='Dense_Layer_8')(x)
        x = layers.Reshape(self.shape_before_flattening,
                           name='Reshape_Layer')(x)
        x = self.decoder_block(inputs=x, filters=256, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=9, use_bn=True)
        x = self.decoder_block(inputs=x, filters=128, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=10, use_bn=True)
        x = self.decoder_block(inputs=x, filters=64, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=11, use_bn=True)
        x = self.decoder_block(inputs=x, filters=32, kernel_size=3,
                               strides=2, padding='same', activation='relu', layer=12, use_bn=True)
        x = self.decoder_block(inputs=x, filters=3, kernel_size=3,
                               strides=2, padding='same', activation='tanh', layer=13, use_bn=False)
        self.decoder = tf.keras.Model(inputs, x, name='decoder')

    def build_vae(self):
        self.build_encoder()
        self.build_decoder()

    @property
    def metrics(self):
        return [
            self.train_total_loss_tracker,
            self.train_reconstruction_loss_tracker,
            self.train_kl_loss_tracker,
            self.test_total_loss_tracker,
            self.train_reconstruction_loss_tracker,
            self.test_kl_loss_tracker
        ]

    def compute_loss(self, inputs, outputs, mu, log_var):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(inputs - outputs), axis=(1, 2, 3)))
        # reconstruction_loss *= 0.1  # reconstruction_loss_factor

        kl_loss = -0.5 * (1 + log_var - tf.exp(log_var) - tf.square(mu))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = kl_loss + reconstruction_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(
                data, reconstruction, mu, log_var)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_total_loss_tracker.update_state(total_loss)
        self.train_reconstruction_loss_tracker.update_state(
            reconstruction_loss)
        self.train_kl_loss_tracker.update_state(kl_loss)

        return {
            'train_total_loss': self.train_total_loss_tracker.result(),
            'train_reconstruction_loss': self.train_reconstruction_loss_tracker.result(),
            'train_kl_loss': self.train_kl_loss_tracker.result()
        }

    def test_step(self, data):
        mu, log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        total_loss, reconstruction_loss, kl_loss = self.compute_loss(
            data, reconstruction, mu, log_var)
        self.test_total_loss_tracker.update_state(total_loss)
        self.test_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.test_kl_loss_tracker.update_state(kl_loss)

        return {
            'total_loss': self.test_total_loss_tracker.result(),
            'reconstruction_loss': self.test_reconstruction_loss_tracker.result(),
            'kl_loss': self.test_kl_loss_tracker.result()
        }

    def sample(self, count=1, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(count, self.latent_dim))
        return self.decoder(eps)
