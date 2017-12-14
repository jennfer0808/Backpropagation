# -*- coding : utf-8 -*-

# ============================================================
#   name : cnn
# author : wzy
#   date : 2017/12/13
#   desc : 
# ============================================================

import numpy as np


# 获取卷积区域
def get_path(input_array, i, j, filter_width, filter_height, stride):
    """
    从输入数组中获取本次卷积的区域，自动适配输入为2D和3D的情况
    :param input_array: 输入
    :param i: 输出的位置
    :param j: 输出的位置
    :param filter_width: 过滤器大小
    :param filter_height: 过滤器大小
    :param stride: 步长
    :return:
    """
    start_i = i * stride
    start_j = j * stride

    if input_array.ndim == 2:
        return input_array[start_i: start_i + filter_height, start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:, start_i: start_i + filter_height, start_j: start_j + filter_width]


# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


def padding(input_array, zero_padding):
    """
    为输入zero padding，自动适配输入为2D和3D的情况
    """
    if zero_padding == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            # 初始化全为0
            padded_array = np.zeros((input_depth, input_height + 2 * zero_padding, input_width + 2 * zero_padding))
            # 拷贝数据
            padded_array[:, zero_padding: zero_padding + input_height, zero_padding:zero_padding + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height + 2 * zero_padding, input_width + 2 * zero_padding))
            padded_array[zero_padding: zero_padding + input_height, zero_padding: zero_padding + input_width] = input_array
            return padded_array


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(input_array, kernel_array, output_array, stride, bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_path(input_array, i, j, kernel_width, kernel_height, stride) * kernel_array).sum() + bias


class Filter(object):
    """
    Filter类保存了巻积层的参数、梯度，用SGD更新参数
    """

    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\n          bias:\n%s'.format(repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvLayer(object):
    def __init__(self, input_width, input_height, input_channel, filter_width, filter_height, filter_numbers, zero_padding, stride, activator, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_numbers = filter_numbers
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding, stride)
        self.output_array = np.zeros((self.filter_numbers, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_numbers):
            self.filters.append(Filter(filter_width, filter_height, self.input_channel))
        self.activator = activator
        self.learning_rate = learning_rate

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_numbers):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f], self.stride, filter.get_bias())
            element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        """
        计算传递给前一层的误差项，以及计算每个权重的梯度。前一层的误差项保存在self.delta_array，梯度保存在Filter对象的weights_grad
        :param input_array:
        :param sensitivity_array: 3D
        :param activator:
        :return:
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        """
        按照梯度下降更新权重
        """
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        """
        计算传递到上一层的sensitivity map
        :param sensitivity_array: 本层的sensitivity map
        :param activator: 上一层的激活函数
        :return:
        """
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)  # 扩展完之后，★★★ 此处的stride=1 ★★★

        # 将expanded_array执行zero padding，并以stride=1的卷积来生成input。
        # ★★★ 此处的隐藏条件是stride=1，并且是由扩充之后的sensitivity map反向卷积到输入层 ★★★
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)

        # 初始化delta_array，用于保存传递到上一层的sensitivity map
        self.delta_array = self.create_delta_array()

        # 对于具有多个filter的卷积层来说，最终传递到上一层的 sensitivity map相当于所有的filter的 sensitivity map之和
        for f in range(self.filter_numbers):
            filter = self.filters[f]

            # 将filter权重翻转180度
            flipped_weights = np.array(map(lambda i: np.rot90(i, 2), filter.get_weights()))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array

        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_numbers):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f], filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array):
        """
        扩展 sensitivity map，在stride>=2的情形，以中间补0的方式，扩充到stride=1的情形。即使stride=1，以下的代码不影响计算
        :param sensitivity_array: 3维的sensitivity map
        :return:
        """
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.input_channel, self.input_height, self.input_width))
