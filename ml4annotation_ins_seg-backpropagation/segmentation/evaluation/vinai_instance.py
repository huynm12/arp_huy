import os
import json
import tempfile
import pickle
import numpy as np
import copy
import time
import torch
import torch._six

import cv2
import numpy as np
from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from .cocoeval_custom import COCOeval
from pycocotools import mask as coco_mask
from panopticapi.utils import id2rgb

from collections import defaultdict
from tqdm import tqdm

from .utils import all_gather
from segmentation.utils import save_annotation

def convert_to_coco_api(ds, instance_pickle=None, panoptic_json=None):
    if os.path.isfile(panoptic_json) and os.path.isfile(instance_pickle):
        return pickle.load(open(instance_pickle, "rb"))

    coco_ds = COCO()

    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    instance_annotations = []
    pan_annotations = []
    categories = set()
    instance_categories = set()
    os.makedirs(os.path.join(ds.root, "annotations/panoptic_val"), exist_ok=True)
    for img_idx in tqdm(range(len(ds))):
        data = ds[img_idx]
        image_id = int(data["id"])
        img_dict = {}
        img_dict['id'] = int(image_id)
        img_dict['height'] = int(data["raw_size"][0])
        img_dict['width'] = int(data["raw_size"][1])
        dataset['images'].append(img_dict)
        panoptic_gt = data["raw_panoptic"].numpy() #cv2.resize(data["panoptic"].numpy(), (data["raw_size"][1], data["raw_size"][0]),
                                            # interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        # For panoptic evaluation
        panoptic_out = np.zeros_like(panoptic_gt)
        pan_annotation = {
            'image_id': image_id,
            'file_name': data["name"] + ".png",
        }
        segments_info = []

        metas = ds.meta_list[img_idx]
        for meta_idx, meta in enumerate(metas):
            
            # For panoptic segmentation
            pan_id = meta["sem_id"] * ds.label_divisor + meta_idx
            panoptic_out[panoptic_gt == meta["index"]] = pan_id
            segments_info.append(
                {
                    'id': pan_id,
                    'category_id': int(meta["sem_id"]),
                    'iscrowd': 0,
                    'area': int(np.sum(panoptic_gt == meta["index"]))
                }
            )
        
            # For instance evaluation
            if not ('ins_id' in meta or 'cat_id' in meta):
                continue
            ann = {}
            ann['image_id'] = image_id
            ann['category_id'] = int(meta.get('cat_id', meta['ins_id']))
            instance_categories.add(ann['category_id'])
            mask = (panoptic_gt == meta["index"]).astype('uint8')
            mask = np.asfortranarray(mask)
            if mask.sum() == 0:
                continue
            ann['area'] = int(mask.sum())
            ann['id'] = ann_id
            ann['iscrowd'] = 0
            ann['segmentation'] = coco_mask.encode(mask)

            pos = np.where(mask)
            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1]))
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0]))
            ann['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            instance_annotations.append(ann)
            ann_id += 1
        
        # Prepare panoptic segmentation
        pan_annotation["segments_info"] = segments_info
        save_annotation(id2rgb(panoptic_out), os.path.join(ds.root, "annotations/panoptic_val"), data["name"], add_colormap=False)
        # cv2.imwrite(os.path.join(ds.root, "annotations/panoptic_val", data["name"] + ".png"), panoptic_out)
        pan_annotations.append(pan_annotation)
    
    
    # Dump for Panoptic evaluation
    with open(os.path.join(ds.root, "annotations/panoptic_val.json"), "w") as f:
        dataset["annotations"] = pan_annotations
        dataset['categories'] = [{'id': i, "isthing": 1 if i in ds.thing_list else 0} for i in range(ds.num_classes)]
        f.write(json.dumps(dataset, indent=4))

    dataset["annotations"] = instance_annotations
    dataset['categories'] = [{'id': i} for i in sorted(instance_categories)]
    
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    with open(os.path.join(ds.root, "annotations/instance_val.pkl"), "wb") as f:
        pickle.dump(coco_ds, f)
    return coco_ds


class VinAIInstanceEvaluator:
    """
    Evaluate VinAI Instance Segmentation
    """
    def __init__(self, dataset, label_divisor, instance_pkl=None, panop_json=None):
        self.iou_types = ["bbox", "segm"]
        self.label_divisor = label_divisor
        print("Preparing dataset...")
        self.coco_gt = convert_to_coco_api(dataset, instance_pkl, panop_json)
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
    
    def update(self, prediction, image_id):
        self.img_ids.append(image_id)

        for iou_type in self.iou_types:
            results = self.prepare(prediction, image_id, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list([image_id])
            _, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, prediction, image_id, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(prediction, image_id)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(prediction, image_id)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, prediction, image_id):
        coco_results = []
        instance_ids = np.unique(prediction)
        for instance_id in instance_ids:
            if instance_id == 0: continue
            mask = (prediction == instance_id).astype('uint8')

            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": instance_id // self.label_divisor,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "score": 1.0,
                }
            )
        return coco_results

    def prepare_for_coco_segmentation(self, prediction, image_id):
        coco_results = []
        instance_ids = np.unique(prediction)
        for instance_id in instance_ids:
            if instance_id == 0: continue

            mask = (prediction == instance_id).astype('uint8')

            rle = coco_mask.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": instance_id // self.label_divisor,
                    "segmentation": rle,
                    "score": 1.0,
                }
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions

def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = coco_mask


def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    Args:
        self (obj): coco object with ground truth annotations
        resFile (str): file name of result file
    Returns:
    res (obj): result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    # import pdb; pdb.set_trace()
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x2 - x1) * (y2 - y1)
            ann['id'] = id + 1
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################