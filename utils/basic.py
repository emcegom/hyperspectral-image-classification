#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
"""
@Time  : 2023/2/28 19:37
@Auth  : emcegom
@Email : emcegom@gmail.com
@File  : basic.py
"""
import cv2
import tensorflow as tf
import numpy as np
import datetime
from enum import Enum
import torch


class Util:
    @staticmethod
    def gauss_kernel_2d(ksize, sigma):

        k = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
        return k @ k.T

    @staticmethod
    def gauss_kernel_2d_torch(kernel_size, sigma, device='cpu'):
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32, device=device)
        center = kernel_size // 2
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

        s = 2 * (sigma ** 2)
        sum_val = 0
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                x = i - center
                y = j - center
                v = torch.tensor(x ** 2 + y ** 2, dtype=torch.float32, device=device)
                kernel[i, j] = torch.exp(- v / s)
                sum_val += kernel[i, j].item()
        kernel /= sum_val
        return kernel

    @staticmethod
    def softmax_np(x, axis=1):
        MAX = np.max(x, axis=axis, keepdims=True)
        x_ = x - MAX
        x_exp = np.exp(x_)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum

    @staticmethod
    def init_tf():
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    @staticmethod
    def cur_day():
        return datetime.datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def cur_time():
        return datetime.datetime.now().strftime("%H-%M-%S")


class DisplayMode(Enum):
    DEFAULT = "0"
    HIGHLIGHT = "1"
    UNDERLINE = "4"
    BLINK = "5"
    INVERSE = "7"
    INVISIBLE = "8"
    UN_HIGHLIGHT = "22"
    UN_UNDERLINE = "24"
    UN_BLINK = "25"
    UN_INVERSE = "27"
    VISIBLE = "28"


class FontColor(Enum):
    BLACK = "30"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    FUCHSIA = "35"
    ULTRAMARINE = "36"
    WHITE = "37"


class BackgroundColor(Enum):
    BLACK = "30"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    FUCHSIA = "35"
    ULTRAMARINE = "36"
    WHITE = "37"


class UtilLog:
    __FONT_PREFIX = "\033["
    __FONT_SUFFIX = "m"

    __FONT_DEFAULT = __FONT_PREFIX + DisplayMode.DEFAULT.value + __FONT_SUFFIX
    __FONT_RED = __FONT_PREFIX + DisplayMode.DEFAULT.value + ";" + FontColor.RED.value + __FONT_SUFFIX
    __FONT_GREEN = __FONT_PREFIX + DisplayMode.DEFAULT.value + ";" + FontColor.GREEN.value + __FONT_SUFFIX
    __FONT_BLUE = __FONT_PREFIX + DisplayMode.DEFAULT.value + ";" + FontColor.BLUE.value + __FONT_SUFFIX

    __TAG = "="
    __TAG_LEN = 20
    __NEED_LOG = True

    def __init__(self, need_log: bool = True):
        self.__NEED_LOG = need_log

    @property
    def need_log(self):
        return self.__NEED_LOG

    @need_log.setter
    def need_log(self, need_log: bool):
        self.__NEED_LOG = need_log

    def __info(self, info: str) -> None:
        print((self.__FONT_GREEN + "{}" + self.__FONT_DEFAULT).format(info))

    def __warn(self, warn: str) -> None:
        print((self.__FONT_RED + "{}" + self.__FONT_DEFAULT).format(warn))

    def __content(self, content: str) -> None:
        print((self.__FONT_BLUE + "{}" + self.__FONT_DEFAULT).format(content))

    def time_stamp(self, tag: str = None) -> None:
        if self.__NEED_LOG is False: return
        stamp_info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ""
        if tag is not None:
            stamp_info = stamp_info + " " + tag
        self.__warn(self.__TAG * self.__TAG_LEN + " %s " % stamp_info + self.__TAG * self.__TAG_LEN)

    def log(self, *args, **kwargs):
        if self.__NEED_LOG is False: return
        self.time_stamp(tag=kwargs.get("tag"))
        for arg in args:
            self.__info("{}".format(arg))
        for key, value in kwargs.items():
            if key == "tag": continue
            if type(value) is np.ndarray:
                self.__info("{} --> shape: {}, dtype: {}".format(key, value.shape, value.dtype))
            elif type(value) is torch.Tensor:
                self.__info(
                    "{} --> shape: {}, dtype: {}, device: {}".format(key, value.shape, value.dtype, value.device))
            elif type(value) is dict:
                self.__info("{} :".format(key))
                for k, v in value.items():
                    self.__info("{} --> {}".format(k, v))
            else:
                self.__info("{} --> {}".format(key, value))
