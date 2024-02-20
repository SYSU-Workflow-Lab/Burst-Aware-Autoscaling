# 数据处理与对应的正则化方法
import numpy as np

def local_scale(x, inner_pivot_element):
    """
    本地正则化方法，参考ES-RNN from paper
    A hybrid method of Exponential Smoothing and Recurrent Neural Networks for time series forecasting
    :param x: 输入参数
    :param inner_pivot_element: pivot值
    :return: x 返回正则化后的结果
    """
    x = x / inner_pivot_element
    # ! 问题：pivot可能特别大，此时seq_x和seq_y会非常接近于0，经过log后会是一个很大的负数
    # * 一个考虑：使用log(x+1)而不是log(x)，但这是否会导致另外的偏向
    # * 另一个考虑：使用对称的，仅用log()
    x = np.log(x + 1)
    return x