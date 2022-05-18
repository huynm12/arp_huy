from functools import partial
import argparse
import os
import pprint
import logging
import time

import cv2
import numpy as np
import torch
from torch import nn
# import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config as glob_config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import comm
from segmentation.solver import build_optimizer, build_lr_scheduler
from segmentation.data import build_train_loader_from_cfg, build_test_loader_from_cfg
from segmentation.solver import get_lr_group_id
from segmentation.utils import save_debug_images
from segmentation.utils import AverageMeter
from segmentation.utils.utils import get_loss_info_str, to_cuda, get_module
from segmentation.evaluation import COCOPanopticEvaluator
from segmentation.modelpost_processing import get_semantic_segmentation, get_panoptic_segmentation, get_instance_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(glob_config, args)

    return args

def train(config):
    
    logger = logging.getLogger('segmentation')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=glob_config.OUTPUT_DIR, distributed_rank=0)
    
    # Update hyperparamters 
    glob_config.defrost()
    glob_config.LOSS.SEMANTIC.WEIGHT = config["semantic_weight"]
    glob_config.LOSS.CENTER.WEIGHT = config["center_weight"]
    glob_config.LOSS.OFFSET.WEIGHT = config["offset_weight"]
    glob_config.freeze()

    logger.info(config)

    # cudnn related setting
    # cudnn.benchmark = glob_config.CUDNN.BENCHMARK
    # cudnn.deterministic = glob_config.CUDNN.DETERMINISTIC
    # cudnn.enabled = glob_config.CUDNN.ENABLED
    device = torch.device('cuda:0')

    # build model
    distributed = False
    model = build_segmentation_model_from_cfg(glob_config)

    model = model.to(device)

    train_data_loader = build_train_loader_from_cfg(glob_config)
    val_data_loader = build_test_loader_from_cfg(glob_config)

    optimizer = build_optimizer(glob_config, model)
    lr_scheduler = build_lr_scheduler(glob_config, optimizer)


    start_iter = 0
    max_iter = glob_config.TRAIN.MAX_ITER
    best_param_group_id = get_lr_group_id(optimizer)

    # initialize model
    if os.path.isfile(glob_config.MODEL.WEIGHTS):
        model_weights = torch.load(glob_config.MODEL.WEIGHTS, map_location='cpu')
        get_module(model, distributed).load_state_dict(model_weights, strict=False)
        logger.info('Pre-trained model from {}'.format(glob_config.MODEL.WEIGHTS))
    elif not glob_config.MODEL.BACKBONE.PRETRAINED:
        if os.path.isfile(glob_config.MODEL.BACKBONE.WEIGHTS):
            pretrained_weights = torch.load(glob_config.MODEL.BACKBONE.WEIGHTS, map_location='cpu')
            get_module(model, distributed).backbone.load_state_dict(pretrained_weights, strict=False)
            logger.info('Pre-trained backbone from {}'.format(glob_config.MODEL.BACKBONE.WEIGHTS))
        else:
            logger.info('No pre-trained weights for backbone, training from scratch.')

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    panoptic_metric = COCOPanopticEvaluator(
        output_dir=os.path.join(glob_config.OUTPUT_DIR, glob_config.TEST.PANOPTIC_FOLDER),
        train_id_to_eval_id=val_data_loader.dataset.train_id_to_eval_id(),
        label_divisor=val_data_loader.dataset.label_divisor,
        void_label=val_data_loader.dataset.label_divisor * val_data_loader.dataset.ignore_label,
        gt_dir=glob_config.DATASET.ROOT,
        split=glob_config.DATASET.TEST_SPLIT,
        num_classes=val_data_loader.dataset.num_classes
    )
    image_filename_list = [
        os.path.splitext(os.path.basename(ann))[0] for ann in val_data_loader.dataset.ann_list]

    # Train loop.
    for i in range(start_iter, max_iter):
        
        model.train()
        for batch_idx, data in enumerate(train_data_loader):
            # data
            start_time = time.time()
            data = to_cuda(data, device)
            data_time.update(time.time() - start_time)

            image = data.pop('image')
            out_dict = model(image, data)
            loss = out_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Get lr.
            lr = optimizer.param_groups[best_param_group_id]["lr"]

            batch_time.update(time.time() - start_time)
            loss_meter.update(loss.detach().cpu().item(), image.size(0))

            if batch_idx == 0 or (batch_idx + 1) % glob_config.PRINT_FREQ == 0:
                msg = '[{0}/{1}][{2}/{3}] LR: {4:.7f}\t' \
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                        i + 1, max_iter, batch_idx + 1, len(train_data_loader), lr, batch_time=batch_time, data_time=data_time)
                msg += get_loss_info_str(get_module(model, distributed).loss_meter_dict)
                logger.info(msg)
        
        model.eval()
        val_loss = 0.0
        val_steps = 0
        panoptic_metric.reset()

        for batch_idx, data in enumerate(val_data_loader):
            for key in data.keys():
                    try:
                        data[key] = data[key].to(device)
                    except:
                        pass
            image = data.pop('image')
            torch.cuda.synchronize(device)
            with torch.no_grad():
                out_dict = model(image, data)
            torch.cuda.synchronize(device)

            loss = out_dict['loss']
            val_loss += loss.cpu().numpy()
            val_steps += 1

            if 'semantic' in out_dict:
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])
            else: 
                semantic_pred = None

            if 'foreground' in out_dict:
                foreground_pred = get_semantic_segmentation(out_dict['foreground'])
            else:
                foreground_pred = None
            
            panoptic_pred, center_pred = get_panoptic_segmentation(
                semantic_pred,
                out_dict['center'],
                out_dict['offset'],
                thing_list=val_data_loader.dataset.thing_list,
                label_divisor=val_data_loader.dataset.label_divisor,
                stuff_area=glob_config.POST_PROCESSING.STUFF_AREA,
                void_label=(
                        val_data_loader.dataset.label_divisor *
                        val_data_loader.dataset.ignore_label),
                threshold=glob_config.POST_PROCESSING.CENTER_THRESHOLD,
                nms_kernel=glob_config.POST_PROCESSING.NMS_KERNEL,
                top_k=glob_config.POST_PROCESSING.TOP_K_INSTANCE,
                foreground_mask=foreground_pred)
            torch.cuda.synchronize(device)

            if panoptic_pred is not None:
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

            image_size = data['size'].squeeze(0).cpu().numpy()
            if panoptic_pred is not None:
                panoptic_pred = panoptic_pred[:image_size[0], :image_size[1]]
            
            # Resize back to the raw image size.
            raw_image_size = data['raw_size'].squeeze(0).cpu().numpy()
            if raw_image_size[0] != image_size[0] or raw_image_size[1] != image_size[1]:
                if panoptic_pred is not None:
                    panoptic_pred = cv2.resize(panoptic_pred.astype('float'),
                                                (raw_image_size[1], raw_image_size[0]),
                                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
            
            panoptic_metric.update(panoptic_pred,
                                    image_filename=image_filename_list[batch_idx],
                                    image_id=int(data['id'][0]))
            if batch_idx % 20 == 0:
                logger.info('Evaluation: [{}/{}]'.format(batch_idx + 1, len(val_data_loader)))
        panoptic_results = panoptic_metric.evaluate()

        with tune.checkpoint_dir(i) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=panoptic_results['panoptic_seg']['PQ'])

        lr_scheduler.step()
    
    logger.info("Training finished.")

def main(num_samples=20, gpus_per_trial=1):
    args = parse_args()
    hyper_params = {
        "semantic_weight": tune.loguniform(0.01, 1.0),
        "center_weight": tune.loguniform(0.01, 200),
        "offset_weight": tune.loguniform(0.01, 1.0)
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    result = tune.run(
        partial(train, ),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=hyper_params,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    main(num_samples=40, gpus_per_trial=1)