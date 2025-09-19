#!/usr/bin/env python3
"""
Variational Autoencoder
"""
import tensorflow.keras as keras
import tensorflow as tf


def sampling(args):
    """Reparameterization trick by sampling from N(mu, sigma^2)."""
    mu, log_var = args
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return mu + tf.exp(0.5 * log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder: encoder model (outputs latent, mean, log variance)
        decoder: decoder model
        auto: full autoencoder model
    """
    # -------- Encoder --------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Mean and log variance
    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    # Latent space with reparameterization
    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(inputs, [z, mu, log_var], name="encoder")

    # -------- Decoder --------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # -------- Autoencoder --------
    z, mu, log_var = encoder(inputs)
    reconstructed = decoder(z)
    auto = keras.Model(inputs, reconstructed, name="autoencoder")

    # -------- Loss Function --------
    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, reconstructed)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)

    # KL divergence
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)

    # VAE loss
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto

