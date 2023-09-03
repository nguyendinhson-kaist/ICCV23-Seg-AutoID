# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_flip
from mmdet.evaluation.functional import bbox_overlaps


@MODELS.register_module()
class DetInstanceTTAModel(BaseTTAModel):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.

    Examples:
        >>> tta_model = dict(
        >>>     type='DetInstanceTTAModel',
        >>>     tta_cfg=dict(nms=dict(
        >>>                     type='nms',
        >>>                     iou_threshold=0.5),
        >>>                     max_per_img=100))
        >>>
        >>> tta_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>          backend_args=None),
        >>>     dict(
        >>>         type='TestTimeAug',
        >>>         transforms=[[
        >>>             dict(type='Resize',
        >>>                  scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>         ], [
        >>>             dict(type='RandomFlip', prob=1.),
        >>>             dict(type='RandomFlip', prob=0.)
        >>>         ], [
        >>>             dict(
        >>>                 type='PackDetInputs',
        >>>                 meta_keys=('img_id', 'img_path', 'ori_shape',
        >>>                         'img_shape', 'scale_factor', 'flip',
        >>>                         'flip_direction'))
        >>>         ]])]
    """

    def __init__(self, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg

    def merge_aug_bboxes(self, aug_bboxes: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            ori_shape = img_info['ori_shape']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            if flip:
                bboxes = bbox_flip(
                    bboxes=bboxes,
                    img_shape=ori_shape,
                    direction=flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores
        
    def merge_aug_masks(self, aug_masks: List[Tensor], img_metas: List[dict]):
        recovered_masks = []
        for mask, img_info in zip(aug_masks, img_metas):
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            if flip:
                if flip_direction == 'horizontal':
                    mask = torch.flip(mask, dims=[-1])
                elif flip_direction == 'vertical':
                    mask = torch.flip(mask, dims=[-2])
                elif flip_direction == 'diagonal':
                    mask = torch.flip(mask, dims=[-1, -2])
                else:
                    raise ValueError(
                        f"Invalid flipping direction '{flip_direction}'")
            recovered_masks.append(mask)
        masks = torch.cat(recovered_masks, dim=0)
        return masks

    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge batch predictions of enhanced data.

        Args:
            data_samples_list (List[List[DetDataSample]]): List of predictions
                of all enhanced data. The outer list indicates images, and the
                inner list corresponds to the different views of one image.
                Each element of the inner list is a ``DetDataSample``.
        Returns:
            List[DetDataSample]: Merged batch prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            data_samples (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            List[DetDataSample]: Merged prediction.
        """
        aug_bboxes = []
        aug_masks = []
        aug_scores = []
        aug_labels = []
        img_metas = []
        # TODO: support instance segmentation TTA
        # assert data_samples[0].pred_instances.get('masks', None) is None, \
        #     'TTA of instance segmentation does not support now.'
        for data_sample in data_samples:
            aug_bboxes.append(data_sample.pred_instances.bboxes)
            aug_masks.append(data_sample.pred_instances.masks)
            aug_scores.append(data_sample.pred_instances.scores)
            aug_labels.append(data_sample.pred_instances.labels)
            img_metas.append(data_sample.metainfo)
            # print(data_sample.pred_instances.bboxes.shape)
            # print(data_sample.pred_instances.masks.shape)
            # print(data_sample.pred_instances.labels.shape)

        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_masks = self.merge_aug_masks(aug_masks, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0)

        if merged_bboxes.numel() == 0:
            return data_samples[0]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.tta_cfg.nms)

        det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]
        det_masks = merged_masks[keep_idxs][:self.tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        _det_masks = det_masks.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.masks = _det_masks
        results.scores = _det_bboxes[:, -1]
        results.labels = det_labels

        results = self._filter_invalid_preds(results)

        det_results = data_samples[0]
        det_results.pred_instances = results
        # print(results.bboxes.shape)
        # print(results.masks.shape)
        # print(results.labels.shape)
        return det_results
    
    def _filter_invalid_preds(self, results: InstanceData, score_thr: List[float] = [0.0, 0.2]) -> InstanceData:
        '''Filter invalid prediction using some inductive priors - VIPriors Challenge 2023'''
        bboxes = results.bboxes
        masks = results.masks
        scores = results.scores
        labels = results.labels

        filtered_bboxes = []
        filtered_masks = []
        filtered_scores = []
        filtered_labels = []

        # process human predictions
        human_thr = score_thr[0]
        human_idx = labels == 0
        human_bboxes = bboxes[human_idx]
        human_masks = masks[human_idx]
        human_scores = scores[human_idx]
        human_labels = labels[human_idx]

        if human_bboxes.numel() != 0:
            filtered_bboxes.append(human_bboxes[human_scores > human_thr])
            filtered_masks.append(human_masks[human_scores > human_thr])
            filtered_scores.append(human_scores[human_scores > human_thr])
            filtered_labels.append(human_labels[human_scores > human_thr])

        # process ball predictions
        ball_thr = score_thr[1]
        ball_idx = labels == 1
        ball_bboxes = bboxes[ball_idx]
        ball_masks = masks[ball_idx]
        ball_scores = scores[ball_idx]
        ball_labels = labels[ball_idx]

        if ball_bboxes.numel() == 0 or ball_scores[0] < ball_thr:
            return results

        for i in range(ball_bboxes.shape[0]):
            if i == 0 or\
                bbox_overlaps(ball_bboxes[i:i+1].cpu().numpy(), ball_bboxes[0:1].cpu().numpy())[0, 0] > 0:
                 
                filtered_bboxes.append(ball_bboxes[i:i+1])
                filtered_masks.append(ball_masks[i:i+1])
                filtered_scores.append(ball_scores[i:i+1])
                filtered_labels.append(ball_labels[i:i+1])
        
        filtered_bboxes = torch.cat(filtered_bboxes, dim=0)
        filtered_masks = torch.cat(filtered_masks, dim=0)
        filtered_scores = torch.cat(filtered_scores, dim=0)
        filtered_labels = torch.cat(filtered_labels, dim=0)

        filter_results = InstanceData()
        filter_results.bboxes = filtered_bboxes
        filter_results.masks = filtered_masks
        filter_results.scores = filtered_scores
        filter_results.labels = filtered_labels
        
        return filter_results
