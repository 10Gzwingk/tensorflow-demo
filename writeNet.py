import mnist
import numpy
numpy.set_printoptions(linewidth=400)
kernel_size = 3
input_size = 28
conv_out_size = (input_size - kernel_size + 1) * (input_size - kernel_size + 1)
conv3x3 = (numpy.random.random((kernel_size, kernel_size)) - 0.5) * 100
fc_1_layer = (numpy.random.random((conv_out_size, conv_out_size)) - 0.5) * 100
fc_2_layer = (numpy.random.random((10, conv_out_size)) - 0.5) * 100


def func(x):
    conv_result = numpy.zeros(shape=[input_size - kernel_size + 1, input_size - kernel_size + 1])

    # conv
    for i in range(input_size - kernel_size):
        for j in range(input_size - kernel_size):
            conv_result[i][j] = numpy.dot(x[i:i + kernel_size, j:j + kernel_size], conv3x3).sum()

    # full connect
    fc_1_input = conv_result.reshape(-1, 1)
    fc_1_output = numpy.matmul(fc_1_layer, fc_1_input)
    fc_2_input = numpy.maximum(fc_1_output, 0)
    fc_2_output = numpy.matmul(fc_2_layer, fc_2_input)
    fc_output = numpy.maximum(fc_2_output, 0)
    return fc_output / numpy.linalg.norm(fc_output)


mnist.temporary_dir = lambda: "E:\\Program\\mnist"
images = mnist.train_images()
for i in range(10):
    print(images[i])
print(images.shape)
print(func(images[0]))
