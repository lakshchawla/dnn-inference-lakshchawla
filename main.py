import tensorflow as tf
import numpy as np
import json
import os

os.makedirs("model", exist_ok=True)

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, validation_split=0.1)

weights = {}
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if w:
        weights[f"W{i}"] = w[0]
        weights[f"b{i}"] = w[1]
np.savez("model/fashion_mnist.npz", **weights)

model_arch = []
for i, layer in enumerate(model.layers):
    config = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": layer.get_config(),
        "weights": [f"W{i}", f"b{i}"] if layer.get_weights() else []
    }
    model_arch.append(config)

with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_arch, f, indent=2)

print("completed")
