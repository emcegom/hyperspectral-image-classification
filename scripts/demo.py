from utils import HSIProcess, HSIVisual, HSIConfigFactory, HSIEntity


def demo():
    entity = HSIProcess.load_dataset(hsi_config=HSIConfigFactory.KSC())
    hsi_pca = HSIProcess.impl_pca(hsi=entity.hsi, n_comp=10)
    hsi_patches, gt_patches = HSIProcess.generate_patches(hsi=hsi_pca, gt=entity.gt, patch_size=7)
    x_train, x_val, x_test, y_train, y_val, y_test = HSIProcess.split_dataset_by_ratio(x=hsi_patches, y=gt_patches,
                                                                                       test_ratio=0.7,
                                                                                       val_ratio_of_train=0.5)
