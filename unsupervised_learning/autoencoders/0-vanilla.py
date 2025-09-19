#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer
            in the encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model (compiled)
    """

    # -------- Encoder --------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(inputs, latent, name="encoder")

    # -------- Decoder --------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # -------- Autoencoder (encoder + decoder) --------
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

