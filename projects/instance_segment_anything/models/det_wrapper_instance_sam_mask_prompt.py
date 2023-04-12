import cv2
import torch

from mmdet.core import bbox2result
from mmdet.models import DETECTORS
from .det_wrapper_instance_sam import DetWrapperInstanceSAM


@DETECTORS.register_module()
class DetWrapperInstanceSAMMaskPrompt(DetWrapperInstanceSAM):
    def __init__(self,
                 stage_2_with_box_p=False,
                 stage_1_multi_mask=False,

                 det_wrapper_type='hdetr',
                 det_wrapper_cfg=None,
                 det_model_ckpt=None,
                 num_classes=80,
                 model_type='vit_b',
                 sam_checkpoint=None,
                 use_sam_iou=True,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DetWrapperInstanceSAMMaskPrompt, self).__init__(det_wrapper_type=det_wrapper_type,
                                                              det_wrapper_cfg=det_wrapper_cfg,
                                                              det_model_ckpt=det_model_ckpt,
                                                              num_classes=num_classes,
                                                              model_type=model_type,
                                                              sam_checkpoint=sam_checkpoint,
                                                              use_sam_iou=use_sam_iou,
                                                              init_cfg=init_cfg,
                                                              train_cfg=train_cfg,
                                                              test_cfg=test_cfg)
        # whether stage 2 input box prompt
        self.stage_2_with_box_p = stage_2_with_box_p
        self.stage_1_multi_mask = stage_1_multi_mask

    def simple_test(self, img, img_metas, rescale=True):
        """Test without augmentation.
        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.
        """
        assert rescale
        assert len(img_metas) == 1
        # results: List[dict(scores, labels, boxes)]
        results = self.det_model.simple_test(img,
                                             img_metas,
                                             rescale)

        # Tensor(n,4), xyxy, ori image scale
        output_boxes = results[0]['boxes']

        image_path = img_metas[0]['filename']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(output_boxes, image.shape[:2])

        # mask_pred: n,1/3,h,w
        # sam_score: n, 1/3
        # coarse_mask: n,1/3,256,256
        _1, coarse_mask_score, coarse_mask = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self.stage_1_multi_mask,
            return_logits=True,
        )
        if self.stage_1_multi_mask:
            max_iou_idx = torch.max(coarse_mask_score, dim=1)[1]
            coarse_mask = (coarse_mask[torch.arange(coarse_mask.size(0)),
                                       max_iou_idx]).unsqueeze(1)
        mask_pred, sam_score, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes if self.stage_2_with_box_p else None,
            mask_input=coarse_mask,
            multimask_output=False,
            return_logits=True,
        )

        # Tensor(n,h,w), raw mask pred
        mask_pred = mask_pred.squeeze(1)
        sam_score = sam_score.squeeze(-1)

        # Tensor(n,)
        label_pred = results[0]['labels']

        score_pred = results[0]['scores']

        # mask_pred: Tensor(n,h,w)
        # label_pred: Tensor(n,)
        # score_pred: Tensor(n,)
        # sam_score: Tensor(n,)
        mask_pred_binary = (mask_pred > self.predictor.model.mask_threshold).float()
        if self.use_sam_iou:
            det_scores = score_pred * sam_score
        else:
            # n
            mask_scores_per_image = (mask_pred * mask_pred_binary).flatten(1).sum(1) / (
                    mask_pred_binary.flatten(1).sum(1) + 1e-6)
            det_scores = score_pred * mask_scores_per_image
        # det_scores = score_pred
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = torch.cat([output_boxes, det_scores[:, None]], dim=-1)
        bbox_results = bbox2result(bboxes, label_pred, self.num_classes)
        mask_results = [[] for _ in range(self.num_classes)]
        for j, label in enumerate(label_pred):
            mask = mask_pred_binary[j].detach().cpu().numpy()
            mask_results[label].append(mask)
        output_results = [(bbox_results, mask_results)]

        return output_results
