# ------------------------------------------------------------------------------
# Loads Cityscapes semantic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os
import torch

import json
import numpy as np

from .base_dataset import BaseDataset
from ..transforms import build_transforms, VinAIPanopticTargetGenerator
from .utils import get_color_augmentation

# _THING_CLASSES = ["Person", "Rider", "Car", "Truck", "Bus", "On-rails", "Motorcycle", "Bicycle"]
# _THING_CLASSES = ["Person", "Car", "Truck", "Bus", "On-rails", "Motorcycle", "Bicycle"]
# _SEMANTIC_CLASSES = {
#     "Road": 0,
#     "Bike-lane": 0, # Merge with road
#     "Lane Line": 0, # Merge with road
#     "Road Marking": 0, # Merge with road
#     "Driveway": 0, # Merge with road
#     "Sidewalk": 1,
#     "Traffic-island": 1, # Merge with sidewalk
#     "Building": 2,
#     "Wall": 3,
#     "Fence": 4,
#     "Street-light": 5, # Merge with pole
#     "Pole": 5,
#     "Traffic-light-general": 6,
#     "General-single": 6, # Merge with traffic light
#     "Cyclist-light": 6, # Merge with traffic light
#     "Pedestrian-light": 6, # Merge with traffic light
#     "Other-light": 6, # Merge with traffic light
#     "Traffic-light-counter": 6, # Merge with traffic light
#     "Cross-light": 6, # Merge with traffic light
#     "Plus-light": 6, # Merge with traffic light
#     "Front-traffic-sign": 7,
#     "Vegetation": 8,
#     "Terrain": 9,
#     "Sky": 10,
#     "Person": 11,
#     "Rider": 12,
#     "Car": 13, 
#     "Truck": 14,
#     "Bus": 15,
#     "On-rails": 16,
#     "Motorcycle": 17,
#     "Bicycle": 18,
# }

_THING_CLASSES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
_SEMANTIC_CLASSES = {
    "Road": 0,
    "Bike-lane": 0, # Merge with road
    "Lane Line": 0, # Merge with road
    "Road Marking": 0, # Merge with road
    "Driveway": 0, # Merge with road
    "Sidewalk": 1,
    "Traffic-island": 1, # Merge with sidewalk
    "Building": 2,
    "Wall": 3,
    "Fence": 4,
    "Street-light": 5, # Merge with pole
    "Pole": 5,
    "Traffic-light-general": 6,
    "General-single": 6, # Merge with traffic light
    "Cyclist-light": 6, # Merge with traffic light
    "Pedestrian-light": 6, # Merge with traffic light
    "Other-light": 6, # Merge with traffic light
    "Traffic-light-counter": 6, # Merge with traffic light
    "Cross-light": 6, # Merge with traffic light
    "Plus-light": 6, # Merge with traffic light
    "Front-traffic-sign": 7,
    "Vegetation": 8,
    "Terrain": 9,
    "Sky": 10,
    "Person": 11,
    "Car": 12, 
    "Truck": 13,
    "Bus": 14,
    "Motorcycle": 15,
    "Bicycle": 16,
}

_THING_IDS = [_SEMANTIC_CLASSES[x] for x in _THING_CLASSES]

class VinAIPanoptic(BaseDataset):
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
                 crop_size=(1208, 1920),
                 rescale_size=(512, 1024),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 small_instance_area=0,
                 small_instance_weight=1,
                 **kwargs):
        super(VinAIPanoptic, self).__init__(root, 'train' if is_train else 'test', is_train, crop_size, mirror, min_scale, max_scale,
                                         scale_step_size, mean, std)

        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        # self.thing_list = list(range(1, len(_THING_CLASSES) + 1))
        self.thing_list = _THING_IDS

        self.num_classes = max(list(_SEMANTIC_CLASSES.values())) + 1
        self.label_divisor = 1000
        self.rescale_h, self.rescale_w = rescale_size

        # Get image and annotation list.
        self.img_list, self.ann_list, self.meta_list = self._get_files(info_file)

        self.transform = build_transforms(self, is_train)
        self.target_transform = VinAIPanopticTargetGenerator(len(_THING_CLASSES), self.ignore_label, small_instance_area=small_instance_area,
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
                
                meta_file = os.path.join(self.root, meta_file)
                meta = json.load(open(meta_file, "r"))
                meta = meta["response"]["labelMapping"]
                metas = []
                for k, v in meta.items():
                    if k in _SEMANTIC_CLASSES.keys():
                        if isinstance(v, dict):
                            v = [v]
                        for x in v:
                            if x.get("numPixels", 0) is None or x.get("numPixels", 0) < 1: continue
                            x["name"] = k
                            if k in _THING_CLASSES:
                                x["ins_id"] = _THING_CLASSES.index(k) + 1
                            x["sem_id"] = _SEMANTIC_CLASSES[k]
                            metas += [x]
                if len(metas) == 0: continue
                meta_list += [metas]
                img_list += [os.path.join(self.root, image_file)]
                ann_list += [os.path.join(self.root, annotation_file)]
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
        # raw_label = label.copy()
        if not self.is_train:
            # Do not save this during training
            target = self.target_transform(label.copy(), self.meta_list[index])
            dataset_dict['raw_label'] = target['semantic']
            dataset_dict['raw_panoptic'] = target['panoptic']
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        name = os.path.splitext(os.path.basename(self.ann_list[index]))[0]

        # Apply color augmentation
        if self.is_train:
            image = self.color_aug(image)

        # Apply data augmentation.
        if self.transform is not None:
            image, label = self.transform(image, label.copy())
            # if self.is_train:
            #     label = train_label
        size = image.shape
        dataset_dict['size'] = np.array([size[1], size[2], size[0]])
        dataset_dict['image'] = image
        dataset_dict['name'] = name
        dataset_dict['id'] = index

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
    
    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap