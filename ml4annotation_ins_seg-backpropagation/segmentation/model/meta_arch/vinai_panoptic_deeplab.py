# ------------------------------------------------------------------------------
# Panoptic-DeepLab meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import os
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseSegmentationModel
from segmentation.model.decoder import InstanceDeepLabDecoder
from segmentation.utils import AverageMeter
from segmentation.model.backbone import HRNet_Mscale
from segmentation.modelpost_processing import get_inconsistency_map, get_instance_weight_map


__all__ = ["VinAIPanopticDeepLab"]

SCALES = [0.25, 0.5, 1.0, 1.5]
class VinAIPanopticDeepLab(BaseSegmentationModel):

    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key,
                 atrous_rates, num_classes, semantic_loss, semantic_loss_weight, center_loss, center_loss_weight,
                 offset_loss, offset_loss_weight, out_semantic_channels=None, 
                 pretrained_semantic=None, ignore_semantic_head=False, thing_ids=[], use_semantic=True, freeze_module="", threshold=None, nms_kernel=None, top_k=None, **kwargs):

        decoder = None
        if kwargs.get('has_instance', False):
            decoder = InstanceDeepLabDecoder(in_channels, feature_key, low_level_channels, low_level_key, 
                                         atrous_rates, **kwargs)
        else:
            backbone = None

        super(VinAIPanopticDeepLab, self).__init__(backbone, decoder)

        self.semantic_backbone = HRNet_Mscale(num_classes, None)

        if kwargs.get('has_instance', False):
            
            self.feature_key = feature_key
        
        self.semantic_encoder=None
        if kwargs.get('has_instance', False) and use_semantic:
            self.semantic_encoder = nn.Sequential(
                        nn.Conv2d(len(thing_ids), 64, kernel_size=3, stride=2, padding=1,
                                bias=False),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, out_semantic_channels, kernel_size=3, stride=2, padding=1,
                                bias=False),
                        nn.BatchNorm2d(out_semantic_channels)
                    )

        self.thing_ids = thing_ids
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.top_k = top_k
        self.freeze_module = freeze_module
        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()

        if self.freeze_module != "semantic":
            self.semantic_loss = semantic_loss
            self.semantic_loss_weight = semantic_loss_weight
            self.loss_meter_dict['Semantic loss'] = AverageMeter()
            # self.loss_meter_dict['Semantic sigma'] = AverageMeter()
            # self.register_parameter(name='semantic_sigma', param=nn.Parameter(torch.tensor(1.0, requires_grad=True)))
        else:
            self.semantic_loss = None
            self.semantic_loss_weight = 0
            # self.semantic_sigma = 1

        if kwargs.get('has_instance', False) and self.freeze_module != "instance":
            self.center_loss = center_loss
            self.center_loss_weight = center_loss_weight
            self.offset_loss = offset_loss
            self.offset_loss_weight = offset_loss_weight
            self.loss_meter_dict['Center loss'] = AverageMeter()
            self.loss_meter_dict['Offset loss'] = AverageMeter()
            # self.loss_meter_dict['Center sigma'] = AverageMeter()
            # self.loss_meter_dict['Offset sigma'] = AverageMeter()
            # self.register_parameter(name='center_sigma', param=nn.Parameter(torch.tensor(1.0, requires_grad=True)))
            # self.register_parameter(name='offset_sigma', param=nn.Parameter(torch.tensor(1.0, requires_grad=True)))
        else:
            self.center_loss = None
            self.center_loss_weight = 0
            self.offset_loss = None
            self.offset_loss_weight = 0
            # self.center_sigma = 1
            # self.offset_sigma = 1

        # Initialize parameters.
        self._init_params()
        self.custom_lrs = {"semantic_sigma": 1e-3, "center_sigma": 1e-3, "offset_sigma": 1e-3}

        # Load pretrained semantic
        if os.path.isfile(pretrained_semantic):
            print("Load pretrained semantic segmentation")
            state_dict = torch.load(pretrained_semantic, map_location='cpu')["state_dict"]
            ignore_keys = ["ocr.ocr_gather_head", "ocr.cls_head", "ocr.aux_head"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            new_state_dict = state_dict
            if ignore_semantic_head:
                print("Ignore semantic head")
                new_state_dict = {}
                for k, v in state_dict.items():
                    is_ignore = False
                    for ignore_key in ignore_keys:
                        if k.startswith(ignore_key):
                            is_ignore = True
                            break
                    if not is_ignore: new_state_dict[k] = v
            self.semantic_backbone.load_state_dict(new_state_dict, strict=False)

    def freeze_layers(self):
        if self.freeze_module == "semantic":
            for p in self.semantic_backbone.parameters():
                p.requires_grad = False
            self.semantic_backbone.eval()
        elif self.freeze_module == "instance":
            for p in self.parameters():
                p.requires_grad = False
            for p in self.semantic_backbone.parameters():
                p.requires_grad = True
        elif self.freeze_module == "all":
            for p in self.parameters():
                p.requires_grad = False

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result
    
    def forward(self, x, targets=None):
        input_shape = x.shape[-2:]
        
        # Forward semantic backbone
        if self.freeze_module == "semantic":
            with torch.no_grad():
                out = self.semantic_backbone({"images": x}, SCALES)
        else:
            out = self.semantic_backbone({"images": x}, SCALES)
        

        # Process the prediction
        # Prepare the mask for foreground region: Fg U Fp
        results = {}
        if isinstance(out, dict):
            results = out
        else:
            results['pred'] = out
        results["semantic"] = results["pred"]
        
        # TODO: Detach the prediction
        # sem_pred = results['pred'].detach()
        sem_pred = results['pred']

        if self.center_loss is None:
            if targets is None:
                return results
            else:
                return self.loss(results, targets)
        
        foreground_prob = None
        if True:
            foreground_pred = sem_pred.argmax(1)
            foreground_mask = torch.zeros_like(foreground_pred)
            for thing_id in self.thing_ids:
                foreground_mask[foreground_pred == thing_id] = 1.0
            
            
            if targets and self.training:
                foreground_mask = targets["foreground"].to(foreground_mask.device) + foreground_mask
                foreground_mask[foreground_mask > 1] = 1

            # Apply mask to prediction
            foreground_prob = foreground_mask.unsqueeze(1) * sem_pred.softmax(1)
            
             # Get tensor of thing ids only
            foreground_prob = foreground_prob[:, self.thing_ids, :, :]#.continguous()
        else:

            # One-hot
            foreground_pred = sem_pred.argmax(1)
            foreground_pred[foreground_pred == 12] = 17

            foreground_prob = torch.zeros_like(sem_pred[:, self.thing_ids, :, :])
            for idx, thing_id in enumerate(self.thing_ids):
                foreground_prob[:, idx][foreground_pred == thing_id] = 1.0

        # contract: features is a dict of tensors
        emb = None
        if self.semantic_encoder is not None:
            emb = self.semantic_encoder(foreground_prob)
        features = self.backbone(x, emb)
        pred = self.decoder(features)
        results.update(self._upsample_predictions(pred, input_shape))
        if targets is None:
            return results
        else:
            return self.loss(results, targets)

    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        loss = 0
        
        # Get inconsistency map
        # inconsistency_map = None
        # if 'semantic' in results and 'center' in results and 'offset' in results:
        #     inconsistency_map = []
        #     for i in range(batch_size):
        #         inconsistency_map += [
        #             get_inconsistency_map(results['pred'][i: i+1].argmax(1), 
        #                                   results['center'][i: i+1], 
        #                                   results['offset'][i: i+1], 
        #                                   self.thing_ids, 
        #                                   threshold=self.threshold, 
        #                                   nms_kernel=self.nms_kernel, 
        #                                   top_k=self.top_k)
        #         ]
        #     inconsistency_map = torch.cat(inconsistency_map, dim=0)
        #     # inconsistency_map[inconsistency_map < 1.0] = 0.5

        # Get instance CE map
        instance_ce_map = None
        if 'semantic' in results and 'center' in results and 'offset' in results:
            instance_ce_map = []
            for i in range(batch_size):
                instance_ce_map += [
                    get_instance_weight_map(results['pred'][i: i+1].argmax(1), 
                                            targets['semantic'][i: i+1],
                                            results['center'][i: i+1], 
                                            results['offset'][i: i+1], 
                                            self.thing_ids, 
                                            threshold=self.threshold, 
                                            nms_kernel=self.nms_kernel, 
                                            top_k=self.top_k)
                ]
            instance_ce_map = torch.cat(instance_ce_map, dim=0)

        if targets is not None:

            if self.semantic_loss is not None:
                # Semantic loss

                # Calculate weight
                semantic_weight = targets.get('semantic_weights', None)
                if instance_ce_map is not None:
                    semantic_weight = instance_ce_map.clone()
                    # semantic_weight = inconsistency_map if semantic_weight is None else inconsistency_map * semantic_weight
                    # targets['semantic'][inconsistency_map == 0] = 255

                semantic_loss = self.semantic_loss(results['pred'], targets['semantic'], semantic_weights=semantic_weight)
                if 'pred_05x' in results and 'pred_10x' in results:
                    semantic_loss = semantic_loss  * 0.5 + self.semantic_loss(results['pred_05x'], targets['semantic'], semantic_weights=semantic_weight) * 0.2 + \
                                    self.semantic_loss(results['pred_10x'], targets['semantic'], semantic_weights=semantic_weight) * 0.3
                semantic_loss = semantic_loss * self.semantic_loss_weight
                
                # s = torch.log(self.semantic_sigma**2)
                # semantic_loss = torch.exp(-s) * semantic_loss + s/2
                loss += semantic_loss
                self.loss_meter_dict['Semantic loss'].update(semantic_loss.detach().cpu().item(), batch_size)
                # self.loss_meter_dict['Semantic sigma'].update(self.semantic_sigma.detach().cpu().item(), batch_size)

            # Center loss
            if self.center_loss is not None:
                # Pixel-wise loss weight
                center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
                area = center_loss_weights.sum()
                
                # Do not use inconsistency loss for centers
                if instance_ce_map is not None:
                    center_loss_weights = instance_ce_map.unsqueeze(1) * center_loss_weights

                center_loss = self.center_loss(results['center'], targets['center']) * center_loss_weights
                # safe division
                if area > 0:
                    center_loss = center_loss.sum() / area * self.center_loss_weight
                else:
                    center_loss = center_loss.sum() * 0
                
                # s = torch.log(self.center_sigma**2)
                # center_loss = (torch.exp(-s) * center_loss + s/2)
                loss += center_loss
                self.loss_meter_dict['Center loss'].update(center_loss.detach().cpu().item(), batch_size)
                # self.loss_meter_dict['Center sigma'].update(self.center_sigma.detach().cpu().item(), batch_size)
            
            # Offset loss
            if self.offset_loss is not None:
                # Pixel-wise loss weight
                offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
                area = offset_loss_weights.sum()

                if instance_ce_map is not None:
                    offset_loss_weights = instance_ce_map.unsqueeze(1) * offset_loss_weights

                offset_loss = self.offset_loss(results['offset'], targets['offset']) * offset_loss_weights
                # safe division
                if area > 0:
                    offset_loss = offset_loss.sum() / area * self.offset_loss_weight
                else:
                    offset_loss = offset_loss.sum() * 0
                
                # s = torch.log(self.offset_sigma**2)
                # offset_loss = torch.exp(-s) * offset_loss + s/2
                loss += offset_loss
                self.loss_meter_dict['Offset loss'].update(offset_loss.detach().cpu().item(), batch_size)
                # self.loss_meter_dict['Offset sigma'].update(self.offset_sigma.detach().cpu().item(), batch_size)
        # In distributed DataParallel, this is the loss on one machine, need to average the loss again
        # in train loop.
        results['loss'] = loss
        self.loss_meter_dict['Loss'].update(loss.detach().cpu().item(), batch_size)
        return results
