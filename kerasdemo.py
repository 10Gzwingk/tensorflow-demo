import tensorflow as tf
import numpy as np
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
# net = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights='Z:\\ml\\resnet50_weights_tf_dim_ordering_tf_kernels.h5',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000
# )

# for layer in net.layers:
#     tf.keras.layers.Add
#     layer.trainable = True
#
# image = open('', 'rb').read()
# label = image_labels['']
# net.summary()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# model.add(tf.keras.layers.Conv2D(10, 3, padding='SAME', activation='relu', input_shape=(28, 28, 1)))
input_v = tf.keras.layers.Input((28, 28, 1))
conv1 = tf.keras.layers.Conv2D(20, 3, padding='SAME', activation='relu', name='conv1')(input_v)
conv = tf.keras.layers.Conv2D(20, 3, padding='SAME', activation='relu', name='conv2')(conv1)
conv = tf.keras.layers.add((conv, conv1))
# conv = tf.keras.layers.Conv2D(50, 3, padding='SAME', activation='relu', name='conv3')(conv)
flat = tf.keras.layers.Flatten()(conv)
dense1 = tf.keras.layers.Dense(1024, 'relu')(flat)
dense2 = tf.keras.layers.Dense(1024, 'relu')(dense1)
output = tf.keras.layers.Dense(10, 'softmax')(dense2)
model = tf.keras.Model(input_v, output)
model.summary()
model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
y_train = tf.keras.utils.to_categorical(y_train, 10)
# model.fit(x_train.reshape((-1, 28, 28, 1)), y_train, epochs=3)
# model.save_weights('Z:\\ml\\MyWeights\\my_lenet.h5')


# img = tf.keras.preprocessing.image.load_img('Z:\\ml\\8.jpg', target_size=(28, 28))
# img = tf.keras.preprocessing.image.img_to_array(img)
img = tf.io.read_file('Z:\\ml\\8.jpg')
img = tf.image.decode_jpeg(img, channels=1)
img = tf.reshape(img, (-1, 28, 28))
print(tf.shape(img))
model.load_weights('Z:\\ml\\MyWeights\\my_lenet.h5')
# model.evaluate(x_test, y_test)
y = model.predict(img)
# y = tf.keras.utils.to_ordinal(y)
print(y)
