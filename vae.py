import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking,Reshape,ReLU,Conv1D,Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model


@keras.saving.register_keras_serializable()
class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        ## Select a random normal value to scale with the variance
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.saving.register_keras_serializable()
class Encoder(Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64,
            name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="relu")
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        ## Forward pass for projecting inputs to an intermediate value
        x = self.dense_proj(inputs)
        ## Forward pass for distribution parameters from projected inputs
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        ## Sample a z value from the distribution
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


@keras.saving.register_keras_serializable()
class Decoder(Layer):
    """
    Decodes a sampled latent vector to the original input value with 2 layers
    """

    def __init__(self, original_dim, intermediate_dim=64,
            name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="relu")
        self.dense_output = Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


@keras.saving.register_keras_serializable()
class VariationalAutoEncoder(keras.Model):
    """
    Combines the encoder and decoder into an end-to-end model for training.
    """

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(
                latent_dim=latent_dim,
                intermediate_dim=intermediate_dim
                )
        self.decoder = Decoder(original_dim,
                intermediate_dim=intermediate_dim
                )

    def call(self, inputs):
        """ Variational inference forward pass """
        ## Encoder returns sampled latent variable and the sampled distribution
        z_mean, z_log_var, z = self.encoder(inputs)
        ## Decoder
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
