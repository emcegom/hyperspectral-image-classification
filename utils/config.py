#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
"""
@Time  : 2023/2/28 19:38
@Auth  : emcegom
@Email : emcegom@gmail.com
@File  : config.py
"""

import os
from typing import Dict
import yaml


class HSIConfig:
    __FILE_TYPE = [".mat"]
    __DATASETS_ABS_DIR = os.path.abspath(
        os.path.dirname(__file__)
        + os.path.sep
        + ".."
        # + os.path.sep
        # + ".."
    ) + "\\datasets\\"
    __INFO_FILE_NAME = "info.yaml"

    # def __new__(cls, *args, **kwargs):
    #     if cls.__name__ not in cls.__INSTANCE:
    #         cls.__INSTANCE[cls.__name__] = object.__new__(cls, )
    #     return cls.__INSTANCE[cls.__name__]

    def __init__(self, name: str):
        assert type(name) is str
        self.__info = self.__get_info_of_dataset(name=name)
        assert self.__info is not None
        self.__name = name

    def __get_info_of_dataset(self, name: str = "",
                              file_name: str = __INFO_FILE_NAME) -> dict:
        with open(self.__DATASETS_ABS_DIR + file_name, "r", encoding="utf-8") as file:
            import yaml
            return yaml.load(stream=file.read(), Loader=yaml.FullLoader).get(name)

    def __get_abs_url_of_file(self, file_prefix: str) -> str:
        if self.__info.get("file_suffix") not in self.__FILE_TYPE:
            raise Exception("Type Error, Only Accepted %s", self.__FILE_TYPE)
        return self.__DATASETS_ABS_DIR + file_prefix + self.__info.get("file_suffix")

    @property
    def hsi_name(self):
        return self.__name

    @property
    def hsi_file_url(self):
        return self.__get_abs_url_of_file(file_prefix=self.__info.get("hsi_file_prefix"))

    @property
    def hsi_file_key(self):
        return self.__info.get("hsi_file_key")

    @property
    def gt_file_url(self) -> str:
        return self.__get_abs_url_of_file(file_prefix=self.__info.get("gt_file_prefix"))

    @property
    def gt_file_key(self) -> str:
        return self.__info.get("gt_file_key")

    @property
    def label_name_en(self) -> list:
        return self.__info.get("label_name_en")

    @property
    def label_name_zh(self) -> list:
        return self.__info.get("label_name_zh")

    @property
    def classes_num(self) -> int:
        return len(self.label_name_en) - 1

    @property
    def info(self) -> str:
        return self.hsi_name + \
            ", \nclasses_num = " + \
            str(self.classes_num) + \
            ", \nhsi_url = " + \
            self.hsi_file_url + \
            ", \ngt_url = " + \
            self.gt_file_url + \
            ", \nlabel_name_en = " + \
            str(self.label_name_en) +\
            ", \nlabel_name_en = " + \
            str(self.label_name_zh)


class HSIConfigFactory:
    __INSTANCE = {}

    def __new__(cls, *args, **kwargs):
        raise Exception("Prohibit Instantiation")

    @classmethod
    def __get_instance(cls, name: str = ""):
        if name not in cls.__INSTANCE:
            cls.__INSTANCE[name] = HSIConfig(name=name)
        return cls.__INSTANCE[name]

    @classmethod
    def KSC(cls) -> HSIConfig:
        return cls.__get_instance(name="KSC")

    @classmethod
    def IP(cls) -> HSIConfig:
        return cls.__get_instance(name="IP")

    @classmethod
    def BOTSWANA(cls) -> HSIConfig:
        return cls.__get_instance(name="BOTSWANA")

    @classmethod
    def SALINAS(cls) -> HSIConfig:
        return cls.__get_instance(name="SALINAS")

    @classmethod
    def PAVIA(cls) -> HSIConfig:
        return cls.__get_instance(name="PAVIA")

    @classmethod
    def PAVIA_U(cls) -> HSIConfig:
        return cls.__get_instance(name="PAVIA_U")

    @classmethod
    def HOUSTON_2013(cls) -> HSIConfig:
        return cls.__get_instance(name="HOUSTON_2013")

    @classmethod
    def SANDIEGO(cls) -> HSIConfig:
        return cls.__get_instance(name="SANDIEGO")





