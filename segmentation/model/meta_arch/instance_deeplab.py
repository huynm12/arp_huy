# ------------------------------------------------------------------------------
# Panoptic-DeepLab meta architecture.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseSegmentationModel
from segmentation.model.decoder import InstanceDeepLabDecoder
from segmentation.utils import AverageMeter


__all__ = ["InstanceDeepLab"]


class InstanceDeepLab(BaseSegmentationModel):

    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key,
                 atrous_rates, center_loss, center_loss_weight,
                 offset_loss, offset_loss_weight, in_semantic_channels=None, out_semantic_channels=None, **kwargs):

        decoder = InstanceDeepLabDecoder(in_channels, feature_key, low_level_channels, low_level_key, 
                                         atrous_rates, **kwargs)
        super(InstanceDeepLab, self).__init__(backbone, decoder)

        if in_semantic_channels:
            self.feature_key = feature_key
            self.semantic_encoder = nn.Sequential(
                        nn.Conv2d(in_semantic_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, out_semantic_channels, kernel_size=3, stride=2, padding=1,
                               bias=False),
                        nn.BatchNorm2d(out_semantic_channels)
                    )

        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()

        self.center_loss = center_loss
        self.center_loss_weight = center_loss_weight
        self.offset_loss = offset_loss
        self.offset_loss_weight = offset_loss_weight
        self.loss_meter_dict['Center loss'] = AverageMeter()
        self.loss_meter_dict['Offset loss'] = AverageMeter()

        # Initialize parameters.
        self._init_params()

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

        # contract: features is a dict of tensors
        emb = None
        if hasattr(self, 'semantic_encoder'):
            emb = self.semantic_encoder(targets["foreground_semantic"])
        features = self.backbone(x, emb)

        pred = self.decoder(features)
        results = self._upsample_predictions(pred, input_shape)

        if targets is None:
            return results
        else:
            return self.loss(results, targets)

    def loss(self, results, targets=None):
        batch_size = results['center'].size(0)
        loss = 0
        if targets is not None:

            # Center loss
            # Pixel-wise loss weight
            center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
            center_loss = self.center_loss(results['center'], targets['center']) * center_loss_weights
            # safe division
            if center_loss_weights.sum() > 0:
                center_loss = center_loss.sum() / center_loss_weights.sum() * self.center_loss_weight
            else:
                center_loss = center_loss.sum() * 0
            self.loss_meter_dict['Center loss'].update(center_loss.detach().cpu().item(), batch_size)
            loss += center_loss
            
            # Offset loss
            # Pixel-wise loss weight
            offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
            offset_loss = self.offset_loss(results['offset'], targets['offset']) * offset_loss_weights
            # safe division
            if offset_loss_weights.sum() > 0:
                offset_loss = offset_loss.sum() / offset_loss_weights.sum() * self.offset_loss_weight
            else:
                offset_loss = offset_loss.sum() * 0
            self.loss_meter_dict['Offset loss'].update(offset_loss.detach().cpu().item(), batch_size)
            loss += offset_loss
        # In distributed DataParallel, this is the loss on one machine, need to average the loss again
        # in train loop.
        results['loss'] = loss
        self.loss_meter_dict['Loss'].update(loss.detach().cpu().item(), batch_size)
        return results
