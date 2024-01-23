import tensorflow as tf


def resnet(input_layer, i):
    conv = tf.keras.layers.Conv2D(64, 3, padding='SAME', activation='relu')(input_layer)
    conv = tf.keras.layers.Conv2D(64, 3, padding='SAME', activation='relu')(conv)
    conv = tf.keras.layers.Conv2D(128, 3, padding='SAME', activation='relu')(conv)
    tf.keras.layers.RandomRotation
    return tf.keras.layers.add((input_layer, conv))


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

layer_input = tf.keras.layers.Input((28, 28, 1))
layer = layer_input
for i in range(16):
    layer = resnet(layer, i)
# layer = tf.keras.layers.Conv2D(10, 3, padding='SAME', activation='relu')(layer)
layer = tf.keras.layers.GlobalAveragePooling2D()(layer)
layer = tf.keras.layers.Dense(1024, 'relu')(layer)
layer = tf.keras.layers.Dense(10, 'softmax')(layer)
# layer = tf.keras.layers.Softmax()(layer)
model = tf.keras.Model(layer_input, layer)
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
y_train = tf.keras.utils.to_categorical(y_train, 10)
print(x_train.shape)
print(y_train.shape)
# model.fit(x_train.reshape((-1, 28, 28, 1)), y_train, epochs=2)
# model.save_weights('Z:\\ml\\MyWeights\\my_resnet_8_256.h5')

img = tf.io.read_file('Z:\\ml\\8.jpg')
img = tf.image.decode_jpeg(img, channels=1)
img = tf.reshape(img, (-1, 28, 28))
print(tf.shape(img))
model.load_weights('Z:\\ml\\MyWeights\\my_resnet_8_256.h5')
# model.evaluate(x_test, y_test)
y = model.predict(img)
# y = tf.keras.utils.to_ordinal(y)
print(y)
