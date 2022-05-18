import os
from datasets.base_dataset import BaseDataset, ActiveBaseDataset
import random
import cv2
import constants
import torch
import numpy as np

"""
701 = train: 367, val: 101, test: 233
固定大小: 360x480
12类，全标注，图像没有 bg
"""

#Color_list
colors = []
with open("fixed_color.txt", 'r') as f:
    for line in f.readlines():
        line_split = line.split(';')
        key = line_split[0]
        color = line_split[1]
        color_split = color.split(',')
        color_int = [int(x) for x in color_split]
        colors.append(color_int)

# colors = [[64, 128, 64],
# [192, 0, 128],
# [0, 128, 192],
# [0, 128, 64],
# [128, 0, 0],
# [64, 0, 128],
# [64, 0, 192],
# [192, 128, 64],
# [192, 192, 128],
# [64, 64, 128],
# [128, 0, 192],
# [192, 0, 64],
# [128, 128, 64],
# [192, 0, 192],
# [128, 64, 64],
# [64, 192, 128],
# [64, 64, 0],
# [128, 64, 128],
# [128, 128, 192],
# [0, 0, 192],
# [192, 128, 128],
# [128, 128, 128],
# [64, 128, 192],
# [0, 0, 64],
# [0, 64, 64],
# [192, 64, 128],
# [128, 128, 0],
# [192, 128, 192],
# [64, 0, 64],
# [192, 192, 0],
# [0, 0, 0],
# [64, 192, 0]]

def get_img_target_paths(img_names, img_dir, target_dir):
    img_paths = [os.path.join(img_dir, x) for x in img_names]
    target_paths = [os.path.join(target_dir, x.replace('.jpeg', '.png')) for x in img_names]
    return img_paths, target_paths

class CamVid(BaseDataset):
    def __init__(self, root, split='train', transforms=None):
        """
        :param root: /nfs2/xs/Datasets/CamVid11
        :param split: train, val, test
        :param transforms: transforms.Compose()
        """
        img_dir = os.path.join(root, split)
        target_dir = os.path.join(root, split+'_labels')
        img_names = os.listdir(img_dir)

        img_paths, target_paths = get_img_target_paths(img_names, img_dir, target_dir)
        super().__init__(img_paths, target_paths, transforms)
        self.num_classes = len(colors)
        self.bg_idx = 0.0

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        target = cv2.imread(self.target_paths[index])
        # target[target == self.bg_idx] = constants.BG_INDEX  # 默认忽略掉已经处理过的?

        img = cv2.resize(img, (640, 480))
        target = cv2.resize(target, (640, 480), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            img, target = self.transform(img, target)
        img = torch.from_numpy(img).permute(2, 0, 1)

        target = torch.from_numpy(target)
        # target = target[..., [2, 1,0]]
        target = target.permute(2, 0, 1).contiguous()

        mapping = {tuple(c): t for c, t in zip(colors, range(len(colors)))}
        # print("MAPPPINGGGGG: ", mapping, "\n MAPPPPING_SHAPPEEEEE: ", len(mapping))
        mask = torch.zeros(target.shape[1], target.shape[2], dtype=torch.long)
        for k in mapping:
            # Get all indices for current class
            idx = (target == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
        # print("MASKKKKKKK\n", mask, "MASKKKKKKK SHAPEEEEEEEEE: ", mask.shape)
        # for class_id in range(len(colors)):
        #     class_im = mask[mask==class_id]
        #     print(class_im.shape)
        #     cv2.imwrite()
        return img.float().cuda(), mask.cuda()


class ActiveCamVid(ActiveBaseDataset):
    def __init__(self, root, split='train', init_percent=10, transforms=None):
        """
        :param root: /nfs2/xs/Datasets/CamVid11
        :param split: train, val, test
        :param transforms: transforms.Compose()
        """
        self.len_total_dataset = 369

        img_dir = os.path.join(root, split)
        target_dir = os.path.join(root, split + '_labels')
        img_names = os.listdir(img_dir)

        # img_paths, target_paths = get_img_target_paths(img_names, img_dir, target_dir)
        # super().__init__(img_paths, target_paths, transforms)
        # split data
        label_imgs, unlabel_imgs = self.random_split_train_data(img_names, init_percent)
        label_img_paths, label_target_paths = get_img_target_paths(label_imgs, img_dir, target_dir)
        unlabel_img_paths, unlabel_target_paths = get_img_target_paths(unlabel_imgs, img_dir, target_dir)

        super().__init__(label_img_paths, label_target_paths,
                         unlabel_img_paths, unlabel_target_paths, transforms)

        self.num_classes = len(colors)
        self.bg_idx = 0.0

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        target = cv2.imread(self.target_paths[index])

        img = cv2.resize(img, (640,480))
        target = cv2.resize(target, (640,480), interpolation = cv2.INTER_NEAREST)
        # target[target == self.bg_idx] = constants.BG_INDEX  # 默认忽略掉已经处理过的?

        if self.transform:
            img, target = self.transform(img, target)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.

        target = torch.from_numpy(target)
        # target = target[..., [2, 1, 0]]
        target = target.permute(2, 0, 1).contiguous()

        mapping = {tuple(c): t for c, t in zip(colors, range(len(colors)))}
        mask = torch.zeros(target.shape[1], target.shape[2], dtype=torch.long)
        for k in mapping:
            # Get all indices for current class
            idx = (target == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
        return img.float().cuda(), mask.cuda()
    # def __init__(self, root, split='train', init_percent=10, transforms=None):
    #     img_dir = os.path.join(root, split)
    #     target_dir = os.path.join(root, split+'_labels')
    #     img_names = os.listdir(img_dir)
    #
    #     # split data
    #     label_imgs, unlabel_imgs = self.random_split_train_data(img_names, init_percent)
    #     label_img_paths, label_target_paths = get_img_target_paths(label_imgs, img_dir, target_dir)
    #     unlabel_img_paths, unlabel_target_paths = get_img_target_paths(unlabel_imgs, img_dir, target_dir)
    #
    #     super().__init__(label_img_paths, label_target_paths,
    #                      unlabel_img_paths, unlabel_target_paths, transforms)
    #     self.num_classes = 11
    #     self.bg_idx = 11
    #
    # def __getitem__(self, index):
    #     img = cv2.imread(self.label_img_paths[index])[:, :, ::-1]
    #     target = cv2.imread(self.label_target_paths[index], cv2.IMREAD_ANYDEPTH).astype(int)
    #     target[target == self.bg_idx] = constants.BG_INDEX  # map bg
    #
    #     if self.transforms is not None:
    #         img, target = self.transforms(img, target)
    #
    #     return img, target

    def random_split_train_data(self, img_names, percent=10):
        random.shuffle(img_names)
        init_select_num = 40 if percent == 10 else round(len(img_names) * percent / 100)

        label_imgs, unlabel_imgs = img_names[:init_select_num], img_names[init_select_num:]
        return label_imgs, unlabel_imgs


if __name__ == '__main__':
    from datasets.build_datasets import data_root, data_label_colors
    from utils.vis import plt_img_target

    dt = CamVid(data_root['CamVid'], split='train')
    # dt = ActiveCamVid(data_root['CamVid'], split='train', init_percent=0.3)
    print(len(dt))

    _, label_colors = data_label_colors['CamVid']

    for idx, (img, target) in enumerate(dt):
        plt_img_target(img, target, label_colors)
        print(idx, img.shape, target.shape)
        if idx == 9:
            break
