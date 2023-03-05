#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
"""
@Time  : 2023/2/28 19:38
@Auth  : emcegom
@Email : emcegom@gmail.com
@File  : processing.py
"""
from typing import Dict, TypeVar
import numpy as np
from .config import HSIConfig
from .basic import UtilLog
import os
import scipy.io as sio
import spectral
import imageio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class HSIEntity:
    __LOG = UtilLog(need_log=True)

    def __init__(self, name: str, hsi: np.ndarray, gt: np.ndarray, label_name: list = None):
        super(HSIEntity, self).__init__()
        self.__hsi = hsi
        self.__gt = gt
        self.__name = name
        self.__label_name_list = list(label_name)

        self.__pixel_total_num = self.__gt.shape[0] * self.__gt.shape[1]
        self.__pixel_labeled_num = np.where(self.__gt != 0)[0].shape[0]
        self.__pixel_labeled_category_num = np.unique(self.__gt).shape[0] - 1

        self.__pixel_labeled_category_value_list = sorted(list(np.unique(self.__gt)))
        self.__pixel_labeled_category_value_list.remove(0)

        # gt_val : [num, per in labeled, label_name]
        self.__pixel_labeled_count_dict = {}

        # gt_val : [row, col]
        self.__pixel_labeled_location_dict = {}

        self.__cal_flag = False

    def __cal_pixel_labeled_count_dict(self):
        container = dict.fromkeys(self.__pixel_labeled_category_value_list, 0)
        for pixel in np.ravel(self.__gt):
            if pixel > 0: container[pixel] += 1
        verify = 0
        for (step, key) in enumerate(self.__pixel_labeled_category_value_list):
            self.__pixel_labeled_count_dict[key] = [
                container[key],
                round(container[key] * 100 / self.__pixel_labeled_num, 5)
            ]
            verify += container[key]

            # step + 1 有标签不是从1直接开始的，不能用key
            self.__pixel_labeled_count_dict[key].append(
                self.__label_name_list[step + 1] if self.__label_name_list is not None else None
            )

        assert verify == self.__pixel_labeled_num, "GT Stats Verification Failure!"

    def __cal_pixel_labeled_location_dict(self):
        for value in self.__pixel_labeled_category_value_list:
            self.__pixel_labeled_location_dict[value] = [[], []]
        for row in range(self.__gt.shape[0]):
            for col in range(self.__gt.shape[1]):
                if self.__gt[row][col] != 0:
                    self.__pixel_labeled_location_dict[self.__gt[row][col]][0].append(row)
                    self.__pixel_labeled_location_dict[self.__gt[row][col]][1].append(col)
        for key in self.__pixel_labeled_location_dict.keys():
            verify = len(self.__pixel_labeled_location_dict[key][0])
            assert verify == self.__pixel_labeled_count_dict[key][0], "GT Position Stats Verification Failure!"

    @property
    def name(self):
        return self.__name

    @property
    def hsi(self):
        return self.__hsi

    @property
    def gt(self):
        return self.__gt

    @property
    def label_name(self):
        return self.__label_name_list

    @property
    def pixel_total_num(self):
        return self.__pixel_total_num

    @property
    def pixel_labeled_num(self):
        return self.__pixel_labeled_num

    @property
    def pixel_labeled_category_num(self):
        return self.__pixel_labeled_category_num

    @property
    def pixel_labeled_category_value(self):
        return self.__pixel_labeled_category_value_list

    def pixel_labeled_count(self) -> Dict:
        if len(self.__pixel_labeled_count_dict) == 0:
            self.__cal_pixel_labeled_count_dict()
        return self.__pixel_labeled_count_dict

    def pixel_labeled_location(self) -> Dict:
        if len(self.__pixel_labeled_location_dict) == 0:
            self.__cal_pixel_labeled_location_dict()
        return self.__pixel_labeled_location_dict

    def show_gt_info(self):

        self.__LOG.log(self.__name,
                       tag="Display GT Information",
                       pixel_total_num=self.pixel_total_num,
                       pixel_labeled_num=self.pixel_labeled_num)
        self.__LOG.log("pixel_labeled_count :")
        self.__LOG.log("gt_val \tnum \tlabeled(%) \tlabel_name")
        for key, val in self.pixel_labeled_count().items():
            self.__LOG.log("\t{} \t{} \t{:.5f} \t{}".format(key, val[0], val[1], val[2]))


class HSIProcess:
    __TYPE_COMP = TypeVar("__TYPE_COMP", int, float)
    __LOG = UtilLog(need_log=True)

    @classmethod
    def open_file(cls, url: str):
        _, ext = os.path.splitext(url)
        ext = ext.lower()
        if ext == '.mat':
            # Load Matlab array
            return sio.loadmat(url)
        elif ext == '.tif' or ext == '.tiff':
            # Load TIFF file
            return imageio.imread(url)
        elif ext == '.hdr':
            img = spectral.open_image(url)
            return img.load()
        else:
            raise ValueError("Unknown file format: {}".format(ext))

    @classmethod
    def load_dataset(cls, hsi_config: HSIConfig, data_type: type = None) -> HSIEntity:
        hsi = np.array(cls.open_file(hsi_config.hsi_file_url)[hsi_config.hsi_file_key])
        gt = np.array(cls.open_file(hsi_config.gt_file_url)[hsi_config.gt_file_key])

        if data_type is not None:
            hsi, gt = hsi.astype(data_type), gt.astype(data_type)
        cls.__LOG.log(hsi_config.hsi_name, tag="Loading in Dataset", hsi=hsi, gt=gt)
        return HSIEntity(name=hsi_config.hsi_name, hsi=hsi, gt=gt, label_name=hsi_config.label_name_en)

    @classmethod
    def impl_pca(cls, hsi: np.ndarray, n_comp: __TYPE_COMP, is_draw: bool = False) -> np.ndarray:
        if n_comp <= 0:
            hsi.astype(dtype=np.float32)
            res = (hsi - np.min(hsi)) / (np.max(hsi) - np.min(hsi))
            return np.array(res, dtype=np.float32)
        x = np.reshape(hsi, (-1, hsi.shape[2]))
        _PCA = PCA(n_components=n_comp, whiten=True)
        x_pca = _PCA.fit_transform(x)
        x_pca = np.reshape(x_pca, (hsi.shape[0], hsi.shape[1], x_pca.shape[1]))
        # x_pca = x_pca.astype(np.float32)
        cls.__LOG.log(tag="PCA Processing", hsi_pca=x_pca)
        if is_draw:
            ev = _PCA.explained_variance_ratio_
            plt.figure(figsize=(12, 6))
            plt.plot(np.cumsum(ev))
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            # plt.savefig('PCA_components.png')
            plt.show()
            plt.close()
        return x_pca

    @classmethod
    def impl_pca_without_normalization(cls, hsi: np.ndarray, n_comp: __TYPE_COMP) -> np.ndarray:
        if n_comp <= 0: return hsi
        if type(n_comp) is float:
            n_comp = int(hsi.shape[-1] * n_comp)
        x = hsi.reshape((-1, hsi.shape[-1]))
        x_norm = x - np.mean(x, axis=0)
        sigma = np.cov(x_norm.T)
        [U, _, _] = np.linalg.svd(sigma)
        pca_ = np.dot(x_norm, U[:, 0: n_comp])
        pca_ = np.reshape(pca_, (hsi.shape[0], hsi.shape[1], n_comp))
        cls.__LOG.log(tag="PCA Processing without Normalized", hsi_pca=pca_)
        return pca_

    @classmethod
    def impl_kpca(cls, hsi: np.ndarray, n_comp: float, kernel: str = "poly", eigen_solver='arpack') -> np.ndarray:
        x = np.reshape(hsi, (-1, hsi.shape[2]))
        kpca = KernelPCA(n_components=n_comp, kernel=kernel, eigen_solver=eigen_solver)
        x = kpca.fit_transform(x)
        x = np.reshape(x, (hsi.shape[0], hsi.shape[1], x.shape[1]))
        x = x.astype(np.float32)
        cls.__LOG.log(tag="KPCA Processing", hsi_kpca=x)
        return x

    @classmethod
    def impl_lda(cls, hsi: np.ndarray, gt: np.ndarray) -> np.ndarray:
        lda = LDA()
        x = lda.fit_transform(np.reshape(hsi, (-1, hsi.shape[2])), np.ravel(gt))
        x = np.reshape(x, (hsi.shape[0], hsi.shape[1], x.shape[1]))
        x = x.astype(np.float32)
        cls.__LOG.log(tag="LDA Processing", hsi_lda=x)
        return x

    @classmethod
    def scale_by_normalization(cls, hsi: np.ndarray) -> np.ndarray:
        x = hsi.reshape((-1, hsi.shape[-1]))
        x = (x - np.min(x, 0)) / (np.max(x, 0) - np.min(x, 0))
        return np.array(x.reshape(hsi.shape), dtype=np.float32)

    @classmethod
    def scale_by_regularization(cls, hsi: np.ndarray) -> np.ndarray:
        x = hsi.reshape((-1, hsi.shape[-1]))
        x = (x - np.mean(x, 0)) / np.std(x, 0)
        return x.reshape(hsi.shape)

    @classmethod
    def generate_patches(cls, hsi: np.ndarray, gt: np.ndarray, patch_size: int,
                         padding_mode: str = "symmetric") -> (np.ndarray, np.ndarray):
        bands = hsi.shape[-1]
        categories = np.unique(gt).shape[0] - 1
        margin = int(patch_size / 2)

        padding_data = np.pad(hsi, ((margin, margin), (margin, margin), (0, 0)), mode=padding_mode)

        padding_label = np.pad(gt, ((margin, margin), (margin, margin)), 'constant')

        pixels = np.where(padding_label != 0)
        num = len(pixels[0])
        patches = np.zeros([num, patch_size, patch_size, bands], dtype=hsi.dtype)
        patches_label = np.zeros((num,), dtype=gt.dtype)

        for (i, (row, col)) in enumerate(zip(pixels[0], pixels[1])):
            patches[i, :, :, :] = padding_data[row - margin: row + margin + 1, col - margin:col + margin + 1, :]
            patches_label[i] = padding_label[row, col]

        cls.__LOG.log(tag="Patch Generation", patch_size=patch_size, margin=margin,
                      padding_mode=padding_mode, padding_data=padding_data, padding_label=padding_label,
                      patches=patches, patches_label=patches_label)
        return patches, patches_label

    @classmethod
    def filter_zero_label_pixel_based(cls, hsi: np.ndarray, gt: np.ndarray) -> (np.ndarray, np.ndarray):
        x = np.reshape(hsi, (-1, hsi.shape[-1]))
        y = gt.flatten()
        x_rm_0 = x[y > 0, :]
        y_rm_0 = y[y > 0]
        cls.__LOG.log(tag="Filter Samples with 0 Tag", x_rm_0=x_rm_0, y_rm_0=y_rm_0)
        return x_rm_0, y_rm_0

    @classmethod
    def split_dataset_by_ratio(cls, x: np.ndarray, y: np.ndarray,
                               test_ratio: float, val_ratio_of_train: float = .0,
                               need_one_hot=True, need_3d=False, random_state: int = 345):
        if need_3d:
            x = np.expand_dims(x, -1)
        if need_one_hot:
            y = OneHotEncoder(sparse=False, dtype=np.float32).fit_transform(y.reshape((-1, 1)))

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_ratio,
                                                            random_state=random_state,
                                                            stratify=y)
        x_val, y_val = None, None
        if val_ratio_of_train != 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                              test_size=val_ratio_of_train,
                                                              random_state=random_state,
                                                              stratify=y_train)
        cls.__LOG.log(tag="Data Segmentation (Scale)",
                      test_ratio=test_ratio, train_ratio=(1 - test_ratio) * (1 - val_ratio_of_train),
                      # "(1 - test_ratio) * (1 - val_ratio_of_train)"
                      val_ratio=(1 - test_ratio) * val_ratio_of_train,
                      data_before_split=x, label_before_split=y,
                      x_train=x_train, y_train=y_train,
                      x_val=x_val, y_val=y_val,
                      x_test=x_test, y_test=y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test

    @classmethod
    def split_dataset_by_num(cls, x: np.ndarray, y: np.ndarray,
                             train_num: int,
                             # val_num_from_train: int,
                             need_one_hot=True, need_3d=False, random_state: int = 345):
        # assert val_num_from_train < train_num, "验证集的数目应小于训练集的数目"
        assert train_num > 0, "Number cannot be less than 1"
        if need_3d:
            x = np.expand_dims(x, -1)

        sample_classified = {}

        for x_, y_ in zip(x, y):
            if y_ not in sample_classified.keys():
                sample_classified[y_] = [x_]
            else:
                sample_classified[y_].append(x_)

        dataset = {}
        for (step, (key, val)) in enumerate(sample_classified.items()):
            assert len(val) >= train_num, "The number of training sets is chosen too much"
            np.random.seed(random_state)
            index = np.arange(0, len(val))
            idx_train = list(np.random.choice(a=index, size=train_num, replace=False))
            idx_test = [idx for idx in index if idx not in idx_train]
            train_x = np.array(val, dtype=np.int32)[idx_train, :]
            train_y = np.array(np.full(shape=(len(idx_train),), fill_value=key))
            test_x = np.array(val, dtype=np.int32)[idx_test, :]
            test_y = np.array(np.full(shape=(len(idx_test),), fill_value=key))
            if step == 0:
                dataset['train_x'] = train_x
                dataset['train_y'] = train_y
                dataset['test_x'] = test_x
                dataset['test_y'] = test_y
            else:
                dataset.update({'train_x': np.concatenate((dataset.get('train_x'), train_x), axis=0)})
                dataset.update({'train_y': np.concatenate((dataset.get('train_y'), train_y), axis=0)})
                dataset.update({'test_x': np.concatenate((dataset.get('test_x'), test_x), axis=0)})
                dataset.update({'test_y': np.concatenate((dataset.get('test_y'), test_y), axis=0)})

        if need_one_hot:
            dataset['train_y'] = OneHotEncoder(sparse=False, dtype=np.float32).fit_transform(
                dataset['train_y'].reshape((-1, 1)))
            dataset['test_y'] = OneHotEncoder(sparse=False, dtype=np.float32).fit_transform(
                dataset['test_y'].reshape((-1, 1)))
        cls.__LOG.log(tag="Data segmentation by number",
                      train_x=dataset['train_x'], train_y=dataset['train_y'],
                      test_x=dataset['test_x'], test_y=dataset['test_y'])
        return dataset['train_x'], dataset['test_x'], dataset['train_y'], dataset['test_y']
