#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Args:
        input_dims (tuple): dimensions of the model input (H, W, C)
        filters (list): number of filters for each convolutional layer in encoder
        latent_dims (tuple): dimensions of the latent space representation

    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """

    # -------- Encoder --------
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent space
    shape_before_flatten = x.shape[1:]  # save shape for decoder
    latent = keras.layers.Conv2D(latent_dims[-1], (3, 3),
                                 activation='relu', padding='same')(x)

    encoder = keras.Model(inputs, latent, name="encoder")

    # -------- Decoder --------
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    for i, f in enumerate(reversed(filters)):
        # Last two convolutions need special handling
        if i < len(filters) - 2:
            x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                    padding='same')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        elif i == len(filters) - 2:
            x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                    padding='valid')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        else:  # final layer
            x = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                    activation='sigmoid',
                                    padding='same')(x)

    decoder = keras.Model(latent_inputs, x, name="decoder")

    # -------- Autoencoder (encoder + decoder) --------
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

