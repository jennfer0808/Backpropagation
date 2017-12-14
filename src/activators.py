# -*- coding : utf-8 -*-

# ============================================================
#   name : activators
# author : wzy
#   date : 2017/12/13
#   desc : 激活函数实例类，每个类都定义了正向和反向计算
# ============================================================


import numpy as np


class ReluActivator(object):
    def forward(self, weight_input):
        return max(0, weight_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weight_input):
        return weight_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, weight_input):
        return 1.0 / (1.0 + np.exp(-weight_input))

    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    def forward(self, weight_input):
        return 2.0 / (1 + np.exp(-2 * weight_input)) - 1.0

    def backward(self, output):
        return 1 - output * output
