#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
"""
@Time  : 2023/2/28 19:38
@Auth  : emcegom
@Email : emcegom@gmail.com
@File  : visualizing.py
"""
import matplotlib.pyplot as plt
import numpy as np
from .config import HSIConfig
from .basic import UtilLog
from .processing import HSIProcess
from sklearn.preprocessing import minmax_scale
from matplotlib.patches import Rectangle
from tensorflow.keras.utils import plot_model
from sklearn.metrics import *
from operator import truediv
import time


class HSIVisual:
    __RGB_CHANNEL = [29, 42, 89]
    __RGB_MAP = np.array(
        [
            [0, 0, 0],  # 1
            [255, 0, 0],  # 2
            [255, 187, 197],  # 3
            [255, 165, 0],  # 4
            [255, 255, 0],  # 5
            [255, 215, 0],  # 6
            [0, 255, 0],  # 7
            [192, 255, 62],  # 8
            [0, 0, 255],  # 9
            [0, 191, 255],  # 10
            [0, 255, 255],  # 11
            [160, 32, 240],  # 12
            [171, 130, 255],  # 13
            [224, 255, 255],  # 14
            [255, 228, 255],  # 15
            [255, 250, 205],  # 16
            [230, 230, 250],  # 17
            [255, 106, 106],  # 18
            [144, 238, 144]  # 19
        ]
    )
    __LOG = UtilLog(need_log=True)

    @classmethod
    def plot_util(cls, img, fig_size=(10, 6),
                  dpi: int = 600, save_name="temp.png", save_path='./',
                  need_plt=True, need_save=True):
        plt.figure(figsize=fig_size)
        plt.imshow(img)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        if need_save:
            plt.savefig(save_path + save_name, dpi=dpi)
        if need_plt:
            plt.show()
        plt.close()

    @classmethod
    def rgb_map(cls, class_num, start=0):
        return minmax_scale(cls.__RGB_MAP[start: start + class_num, :], feature_range=(0, 1))

    @classmethod
    def visual_hsi_to_rgb(cls, hsi_config: HSIConfig, rgb_channel=None,
                          dpi=600, fig_size: tuple = (15, 15), save_path='./',
                          need_plt=True, need_save=True):
        if rgb_channel is None:
            rgb_channel = cls.__RGB_CHANNEL

        hsi = HSIProcess.load_dataset(hsi_config=hsi_config).hsi
        x = hsi.reshape([-1, hsi.shape[2]])
        from sklearn.preprocessing import minmax_scale, StandardScaler
        x = StandardScaler().fit_transform(x)
        x_normalized = minmax_scale(x, feature_range=(0.0, 1.0))
        rgb_map = np.reshape(x_normalized, hsi.shape)[:, :, rgb_channel]
        cls.__LOG.log(tag="高光谱图像可视化")
        cls.plot_util(img=rgb_map, fig_size=fig_size, dpi=dpi, save_name="hsi_map.png",
                      need_plt=need_plt, need_save=need_save)

    @classmethod
    def visual_gt_to_cmap(cls, hsi_config: HSIConfig,
                          need_bg: bool = True, dpi: int = 600, fig_size=(15, 15),
                          save_path: str = "./", need_plt: bool = True, need_save: bool = True):
        gt = HSIProcess.load_dataset(hsi_config=hsi_config).gt
        y = gt.reshape([-1, 1]).astype(np.int32)
        # change start to get different color map
        classes_cmap = cls.rgb_map(class_num=hsi_config.classes_num + 1, start=0) if need_bg else cls.rgb_map(
            class_num=hsi_config.classes_num, start=1)
        gt_map = np.zeros(shape=[y.shape[0], 3])
        for i in range(y.shape[0]):
            gt_map[i, :] = classes_cmap[y[i, 0]]
        gt_map = np.reshape(gt_map, [gt.shape[0], gt.shape[1], 3])
        cls.__LOG.log(tag="HSI GT Visualizing")
        cls.plot_util(img=gt_map, fig_size=fig_size, dpi=dpi, save_name="gt_map.png", save_path=save_path,
                      need_plt=need_plt, need_save=need_save)

    @classmethod
    def visual_cbar(cls, hsi_config: HSIConfig,
                    fig_size=(14, 2),
                    cell_width=300,
                    cell_height=30,
                    swatch_width=50,
                    margin=20,
                    top_margin=20,
                    cols=5,
                    empty_cols=0,
                    font_size=20,
                    save_path='./', need_plt=True, need_save=True,
                    need_bg=True, dpi=600):
        gt = HSIProcess.load_dataset(hsi_config=hsi_config).gt
        gt_names = hsi_config.label_name_en
        classes_cmap = cls.rgb_map(class_num=hsi_config.classes_num + 1, start=0) if need_bg else cls.rgb_map(
            class_num=hsi_config.classes_num, start=1)
        cbar = {}
        for i in range(len(gt_names)):
            cbar[gt_names[i]] = classes_cmap[i]

        names = list(cbar)
        n = len(names)
        n_cols = cols - empty_cols
        n_rows = n // n_cols + int(n % n_cols > 0)

        width = cell_width * cols + 2 * margin
        height = cell_height * n_rows + margin + top_margin

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        fig.subplots_adjust(margin / width, margin / height,
                            (width - margin) / width,
                            (height - top_margin) / height)
        ax.set_xlim(0, cell_width * cols)
        ax.set_ylim(cell_height * (n_rows - 0.5), -cell_height / 2.)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_off()
        # ax.set_title(title, fontsize=12, loc="left", pad=10)
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'KaiTi',
                                           'FangSong']
        # Chinese font, preferred to use bold, if it can not be found, then use regular
        plt.rcParams['font.size'] = font_size  # Font Size
        plt.rcParams['axes.unicode_minus'] = False  # The negative sign is displayed normally

        for i, name in enumerate(names):
            row = i % n_rows
            col = i // n_rows
            y = row * cell_height

            swatch_start_x = cell_width * col
            text_pos_x = cell_width * col + swatch_width + 7

            ax.text(text_pos_x, y, name, fontsize=font_size, weight=100,
                    horizontalalignment='left',
                    verticalalignment='center')

            ax.add_patch(
                Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                          height=14, facecolor=cbar[name], edgecolor='1.0')
            )

        if need_save:
            plt.savefig(save_path + 'colormap.png')
        if need_plt:
            plt.show()
        plt.close()

    @classmethod
    def visual_batch_hsi_gt(cls, hsi_configs: dict, save_path='./'):
        assert hsi_configs is not None
        for name, hsi_config in hsi_configs.items():
            cls.visual_hsi_to_rgb(hsi_config=hsi_config, need_save=False, save_path=save_path)
            cls.visual_gt_to_cmap(hsi_config=hsi_config, need_save=False, save_path=save_path)

    @classmethod
    def visual_pred_by_gt_labeled(cls, hsi_config: HSIConfig, model, windowsize, if3D):

        print('Drawing Predicted HSI！！！！！！！！！！！！！！！！')
        entity = HSIProcess.load_dataset(hsi_config=hsi_config)
        hsi = entity.hsi
        gt = entity.gt
        margin = int(windowsize / 2)

        paddingdata = np.pad(hsi, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

        pixels = np.asarray(np.where(gt != 0))
        pixels_pad = pixels + margin

        num_pixels = pixels.shape[1]
        outputs = np.zeros(gt.shape)
        pixel_cubes = np.zeros((num_pixels, windowsize, windowsize, hsi.shape[2]))

        if if3D:
            for i in range(num_pixels):
                pixel_cubes[i, :, :, :] = paddingdata[pixels_pad[0][i] - margin: pixels_pad[0][i] + margin + 1,
                                          pixels_pad[1][i] - margin: pixels_pad[1][i] + margin + 1,
                                          :].reshape(-1, windowsize, windowsize, hsi.shape[2], 1).astype('float32')
        else:
            for i in range(num_pixels):
                pixel_cubes[i, :, :, :] = paddingdata[pixels_pad[0][i] - margin: pixels_pad[0][i] + margin + 1,
                                          pixels_pad[1][i] - margin: pixels_pad[1][i] + margin + 1,
                                          :].reshape(-1, windowsize, windowsize, hsi.shape[2]).astype('float32')

        predictions = np.argmax(model.predict(pixel_cubes), axis=1) + 1

        for i in range(num_pixels):
            outputs[pixels[0][i], pixels[1][i]] = predictions[i]

        pixels_false = np.asarray(np.where(outputs != gt))
        print('The pixel coordinates that are not correctly predicted are：\n%s' % pixels_false)

        print('######################################')

        plt.figure(figsize=(20, 6))
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(gt, cmap='jet')
        plt.axis('off')
        plt.title('Truth Map')
        plt.colorbar(ticks=range(0, 17))

        ax1 = plt.subplot(1, 2, 2)
        plt.imshow(outputs, cmap='jet')
        plt.axis('off')
        plt.title('Predict')
        plt.colorbar(ticks=range(0, 17))
        plt.savefig('predicted_map.png')
        plt.show()

        print('End!!!!')
        return outputs

    @classmethod
    def visual_pred_by_gt_full(cls,
                               hsi_config: HSIConfig,
                               n_component,
                               model_path,
                               window_size,
                               if_3D=False,
                               save_path='./',
                               figsize=(15, 12),
                               dpi=600,
                               start=1
                               ):
        entity = HSIProcess.load_dataset(hsi_config=hsi_config)
        hsi = entity.hsi
        gt = entity.gt
        hsi_pca = HSIProcess.impl_pca(hsi=hsi, n_comp=n_component, is_draw=False)
        pad = window_size // 2
        padding_data = np.pad(hsi_pca, [(pad, pad), (pad, pad), (0, 0)], mode='symmetric')
        padding_data = np.array(padding_data, dtype=np.float32)
        predict = np.zeros(shape=(hsi.shape[0], hsi.shape[1]), dtype=np.float32)
        from tensorflow.keras.models import load_model
        model = load_model(filepath=model_path)
        for i in range(hsi.shape[0]):
            data = []
            for j in range(hsi.shape[1]):
                # need refined
                if if_3D:
                    temp = np.reshape(padding_data[i: i + window_size, j:j + window_size, :],
                                      (window_size, window_size, hsi.shape[-1], 1))
                else:
                    temp = np.reshape(padding_data[i: i + window_size, j:j + window_size, :],
                                      (window_size, window_size, hsi.shape[-1]))
                ###
                data.append(temp)
            batch = np.array(data, dtype=np.float32)
            pred = np.argmax(model.predict(batch), axis=1)
            predict[i, :] = pred
        from scipy.io import savemat, loadmat
        savemat(save_path + 'predict_' + str(entity.name) + '.mat', {'pred': predict.astype(np.int)})

        y = np.reshape(predict, (-1, 1)).astype(np.int)
        cmap = cls.rgb_map(class_num=len(np.unique(predict)) + 1, start=start)

        y_map = np.zeros(shape=[y.shape[0], 3])
        for i in range(y.shape[0]):
            y_map[i, :] = cmap[y[i, 0]]
        y_map = np.reshape(y_map, (predict.shape[0], predict.shape[1], 3))

        plt.figure(figsize=figsize)
        plt.imshow(y_map)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path + "pred.png", dpi=dpi)

    @classmethod
    def model_visual_summary_k(cls, model, path='./', figsize=None):
        model.summary()
        cls.__LOG.log("绘制网络结构")
        plot_model(model=model, to_file=path + "model.png")
        plt.close()

    @classmethod
    def model_plot_metric_k(cls, history, metric):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, val_metrics, 'ro-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.show()
        plt.close()

    @classmethod
    def model_plot_acc_k(cls, history, path='./'):
        plt.figure(figsize=(7, 7))
        plt.rcParams.update({'font.size': 14})
        plt.grid()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='lower right')
        plt.savefig(path + 'tradition_valid_acc.png')
        # plt.show()

        plt.figure(figsize=(7, 7))
        plt.rcParams.update({'font.size': 14})
        plt.grid()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper right')
        plt.savefig(path + 'tradition_valid_loss.png')
        # plt.show()
        plt.close()

    @classmethod
    def model_evaluate_k(cls, model, x_test, y_test, train_time=-1, file_path="./"):
        file_name = file_path + "classification_report.txt"
        start = time.time()
        test_y_pred = model.predict(x_test)
        end = time.time()
        # print('Time of Testing ：%s' % (end - start))

        test_y_pred_arg = np.argmax(test_y_pred, axis=1)
        test_y_arg = np.argmax(y_test, axis=1)

        oa = accuracy_score(test_y_arg, test_y_pred_arg)
        # print('OverallAccuracy：%s' % oa)
        kappa = cohen_kappa_score(test_y_arg, test_y_pred_arg)
        # print('KAPPA：%s' % kappa)
        # score = model.evaluate(test_x, test_y)
        # test_loss = score[0]
        # test_acc = score[1]
        # print('TestLoss: %s , TestAccuracy：%s' % (test_loss, test_acc))
        confusion = confusion_matrix(test_y_arg, test_y_pred_arg)
        each_acc = np.nan_to_num(truediv(np.diag(confusion), np.sum(confusion, axis=1)))
        aa = np.mean(each_acc)
        # print('EachAccuracy: \n{}'.format(each_acc))
        # print('AverageAccuracy: %s ' % aa)

        classification = classification_report(test_y_arg, test_y_pred_arg, target_names=None, digits=5)

        with open(file_name, 'w', encoding="utf-8") as x_file:
            # x_file.write('{} Test loss'.format(test_loss))
            # x_file.write('\n')
            # x_file.write('{} Test accuracy'.format(test_acc))
            # x_file.write('\n')
            # x_file.write('\n')
            x_file.write('{} \n{}\n{}\n OA, AA, Kappa,\n'.format(oa, aa, kappa))
            # x_file.write('\n')
            # x_file.write('{} \nOverall accuracy'.format(oa))
            # x_file.write('\n')
            # x_file.write('{} \nAverage accuracy'.format(aa))
            x_file.write('\n')
            x_file.write('{} train time'.format((train_time)))
            x_file.write('\n')
            x_file.write('{} Test time'.format((end - start)))

            x_file.write('\n')
            x_file.write('{}'.format(each_acc))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            # x_file.write('\n')
            # x_file.write('{}'.format(confusion))


class DrawSpectral:
    def __init__(self, hsi, gt, *args, **kwargs):
        super(DrawSpectral, self).__init__()
        self.hsi, self.gt = hsi, gt
        self.pixel_labeled_local = np.array(np.where(self.gt != 0))
        self.class_num = np.unique(self.gt)[1:]
        self.each_pixel = self.__get_pixel_of_each_class()
        self.all_pixel_mean = None
        self.all_pixel_std = None

    def __get_pixel_of_each_class(self):
        all_pixel = {}
        for i in self.class_num:
            all_pixel.update({i: []})

        for i in range(self.pixel_labeled_local.shape[1]):
            r, c = self.pixel_labeled_local[0][i], self.pixel_labeled_local[1][i]
            pixel_label = self.gt[r, c]
            if pixel_label in self.class_num:
                all_pixel.get(pixel_label).append(self.hsi[r, c, :])
        return all_pixel

    def plot_each_class_pixel(self):
        for k in self.class_num:
            plt.figure(figsize=(12, 6))
            for i in range(len(self.each_pixel.get(k))):
                plt.plot(range(0, self.hsi.shape[2]), np.array(self.each_pixel.get(k)[i]), 'r')
            # plt.legend()
            plt.title(str(k) + 'Class HSI Pxiel signature', fontsize=14)
            plt.xlabel('Band Number', fontsize=14)
            plt.ylabel('Pixel Intensity', fontsize=14)
            plt.savefig('./' + str(k) + 'PixelSpectral.png')
            plt.show()

    def plot_each_class_pixel_avg(self, times=0):
        plt.figure(figsize=(12, 6))
        for k in self.class_num:
            bands = self.hsi.shape[2]
            values_std = np.std(np.array(self.each_pixel.get(k)), axis=0)
            values_mean = np.mean(np.array(self.each_pixel.get(k)), axis=0) + k * times

            plt.plot(range(0, bands), values_mean, label=f'CLass - {k}')
        plt.legend()
        plt.title(f'Pixel class signature', fontsize=14)
        plt.xlabel('Band Number', fontsize=14)
        plt.ylabel('Pixel Intensity', fontsize=14)
        plt.savefig('./PixelSpectral_avg.png')
        plt.show()

    def get_all_labeled_pixel(self):
        all_pixel = np.reshape(self.hsi, (-1, self.hsi.shape[2]))
        self.all_pixel_mean = np.mean(all_pixel, axis=0)
        self.all_pixel_std = np.std(all_pixel, axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(range(0, self.hsi.shape[2]), self.all_pixel_std, 'r', f'CLass - std')
        plt.plot(range(0, self.hsi.shape[2]), self.all_pixel_mean, 'b', f'CLass - mean')
        plt.legend()
        plt.title('All Class HSI Pxiel signature', fontsize=14)
        plt.xlabel('Band Number', fontsize=14)
        plt.ylabel('Pixel Intensity', fontsize=14)
        # plt.savefig(str(k) + 'PixelSpectral.png')
        plt.show()

    def all_pixel_std_sort(self):
        self.get_all_labeled_pixel()
        new_std = {}
        for i in range(self.all_pixel_std.shape[0]):
            new_std.update({i: self.all_pixel_std[i]})
        new_std = zip(new_std.values(), new_std.keys())
        return sorted(new_std)

    def forward(self):
        self.plot_each_class_pixel()
        self.plot_each_class_pixel_avg()
        self.get_all_labeled_pixel()
        self.all_pixel_std_sort()
