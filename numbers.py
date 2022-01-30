# coding=<UTF-8>
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger

logger.setLevel(logging.ERROR)

# Obtenemos set de datos y metadatos del set
dataset, metadata = tfds.load("mnist", as_supervised=True, with_info=True)
# 60.000 datos para enrtrenamiento y 10.000 para validacion
train_dataset, test_dataset = dataset["train"], dataset["test"]

# Definimos etiquetas de texto para cada posible respuesta de la red
class_names = [
    "Cero",
    "Uno",
    "Dos",
    "Tres",
    "Cuatro",
    "Cinco",
    "Seis",
    "Siete",
    "Ocho",
    "Nueve",
]

# Obtenemos la cantidad de ejemplos en variables para utilziarlos despues
num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples

# Normalizar numeros 0-255
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_dataset = train_dataset.map(normalize)
test_dataset = train_dataset.map(normalize)

# Estructura de la red
model = tf.keras.Sequential(
    [
        # Capa de entrada 784 neuronas especificando que llegara en una capa cuadrada de 28x28
        tf.keras.layers.Flatten(imput_shape=(28, 28, 1)),
        # Dos capas ocultas densas de 64 cada una
        tf.keras.Dense(64, activation=tf.nn.relu),
        tf.keras.Dense(64, activation=tf.nn.relu),
        # Capa de salida
        tf.keras.Dense(10, activation=tf.nn.softmax),
    ]
)

# Compilar modelo especificando funcion de costo
# Indicar funciones a utilizar
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Aprendizaje por lotes de 32 cada lote
BATCHSIZE = 32
# Reordenar datos de manera aleatoria en lotes de 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

# Realizar aprendizaje
model.fit(
    # Especificar epocas: numero de vueltas del entrenamiento
    train_dataset,
    epochs=5,
    steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE),
)

# Evaluar el modelo entranado contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / 32)
)

print("Resultado: ", test_accuracy)
