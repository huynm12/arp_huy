# ------------------------------------------------------------------------------
# Loads Cityscapes semantic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os
import torch

import json
import numpy as np

from .base_dataset import BaseDataset
from ..transforms import build_transforms, InstanceTargetGenerator
from .utils import get_color_augmentation

_THING_CLASSES = ["Person", "Car", "Truck", "Bus", "On-rails", "Motorcycle", "Bicycle"]


class VinAIInstance(BaseDataset):
    """
    VinAI instance segmentation dataset.
    Arguments:
        root: Str, root directory.
        info_file: file to be get files from.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
    """
    def __init__(self,
                 root,
                 info_file,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 small_instance_area=0,
                 small_instance_weight=1,
                 **kwargs):
        super(VinAIInstance, self).__init__(root, 'train' if is_train else 'test', is_train, crop_size, mirror, min_scale, max_scale,
                                         scale_step_size, mean, std)

        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        self.thing_list = list(range(1, len(_THING_CLASSES) + 1))
        self.label_divisor = 1000

        # Get image and annotation list.
        self.img_list, self.ann_list, self.meta_list = self._get_files(info_file)

        self.transform = build_transforms(self, is_train)
        self.target_transform = InstanceTargetGenerator(len(_THING_CLASSES), self.ignore_label, small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)
        self.color_aug = get_color_augmentation(0.3)

    def _get_files(self, info_file):
        """Gets files for the specified data type and dataset split.

        Params
        ------
        info_file: txt file to get list of files

        Returns
        -------
            A list of sorted file names or None when getting label for test set.
        """
        info_text = os.path.join(self.root, info_file)
        img_list = []
        ann_list = []
        meta_list = []
        with open(info_text, "r") as f:
            for line in f.readlines():
                _, image_file, annotation_file, meta_file = line.strip().split("\t")
                img_list += [os.path.join(self.root, image_file)]
                ann_list += [os.path.join(self.root, annotation_file)]
                meta_file = os.path.join(self.root, meta_file)
                meta = json.load(open(meta_file, "r"))
                meta = meta["response"]["labelMapping"]
                metas = []
                for k, v in meta.items():
                    if not k in _THING_CLASSES:
                        continue
                    if isinstance(v, dict):
                        v = [v]
                    for x in v:
                        x["name"] = k
                        x["cat_id"] = _THING_CLASSES.index(k) + 1
                        metas += [x]
                meta_list += [metas]
        return img_list, ann_list, meta_list

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        dataset_dict = {}
        assert os.path.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        image = self.read_image(self.img_list[index], 'RGB')
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()
        if self.ann_list is not None:
            assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            label = self.read_label(self.ann_list[index], self.label_dtype)
        else:
            label = None
        raw_label = label.copy()
        if not self.is_train:
            # Do not save this during training
            dataset_dict['raw_label'] = raw_label
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        name = os.path.splitext(os.path.basename(self.ann_list[index]))[0]

        # Apply color augmentation
        if self.is_train:
            image = self.color_aug(image)

        # Apply data augmentation.
        if self.transform is not None:
            image, label = self.transform(image, label)
        size = image.shape
        dataset_dict['size'] = np.array([size[1], size[2], size[0]])
        dataset_dict['image'] = image
        dataset_dict['name'] = name

        # Generate training target.
        label_dict = self.target_transform(label, self.meta_list[index])
        for key in label_dict.keys():
            dataset_dict[key] = label_dict[key]
            # try:
            #     print(name, key, dataset_dict[key].shape)
            # except:
            #     print(name, key, len(dataset_dict[key]))
            #     del dataset_dict[key]
        # import pdb; pdb.set_trace()
        return dataset_dict