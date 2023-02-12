# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow.keras as keras
import mnist
import numpy as numpy

model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), name="digits"),
            keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
        ]
    )
model.summary()


mnist.temporary_dir = lambda: "E:\\Program\\mnist"
images = mnist.train_images().astype("float32") / 255
images = numpy.expand_dims(images, -1)
labels = mnist.train_labels().astype("float32") / 255
labels = numpy.expand_dims(labels, -1)
labels = keras.utils.to_categorical(labels, 10)
print(images.shape)
print(labels.shape)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(images, labels)

test_images = mnist.test_images().astype("float32") / 255
test_labels = mnist.test_labels().astype("float32") / 255
test_images = numpy.expand_dims(test_images, -1)
test_labels = numpy.expand_dims(test_labels, -1)
test_labels = keras.utils.to_categorical(test_labels, 10)
score = model.evaluate(test_images, test_labels, verbose=0)
print(score)

print(test_images[0])
print(model.predict(test_images[:10]))
