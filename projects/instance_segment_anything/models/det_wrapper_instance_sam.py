import cv2
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.core import bbox2result
from mmdet.models import DETECTORS, BaseDetector
from projects.instance_segment_anything.models.segment_anything import sam_model_registry, SamPredictor
from .focalnet_dino.focalnet_dino_wrapper import FocalNetDINOWrapper
from .hdetr.hdetr_wrapper import HDetrWrapper


@DETECTORS.register_module()
class DetWrapperInstanceSAM(BaseDetector):
    wrapper_dict = {'hdetr': HDetrWrapper,
                    'focalnet_dino': FocalNetDINOWrapper}

    def __init__(self,
                 det_wrapper_type='hdetr',
                 det_wrapper_cfg=None,
                 det_model_ckpt=None,
                 num_classes=80,

                 model_type='vit_b',
                 sam_checkpoint=None,
                 use_sam_iou=True,
                 best_in_multi_mask=False,

                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DetWrapperInstanceSAM, self).__init__(init_cfg)
        self.learnable_placeholder = nn.Embedding(1, 1)
        det_wrapper_cfg = Config(det_wrapper_cfg)
        assert det_wrapper_type in self.wrapper_dict.keys()
        self.det_model = self.wrapper_dict[det_wrapper_type](args=det_wrapper_cfg)
        if det_model_ckpt is not None:
            load_checkpoint(self.det_model.model,
                            filename=det_model_ckpt,
                            map_location='cpu')

        self.num_classes = num_classes

        # Segment Anything
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        _ = sam.to(device=self.learnable_placeholder.weight.device)
        self.predictor = SamPredictor(sam)
        # Whether use SAM's predicted IoU to calibrate the confidence score.
        self.use_sam_iou = use_sam_iou
        # If True, set multimask_output=True and return the mask with highest predicted IoU.
        # if False, set multimask_output=False and return the unique output mask.
        self.best_in_multi_mask = best_in_multi_mask

    def init_weights(self):
        pass

    def simple_test(self, img, img_metas, rescale=True, ori_img=None):
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

        if ori_img is None:
            image_path = img_metas[0]['filename']
            ori_img = cv2.imread(image_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(ori_img)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(output_boxes, ori_img.shape[:2])

        # mask_pred: n,1/3,h,w
        # sam_score: n, 1/3
        mask_pred, sam_score, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self.best_in_multi_mask,
            return_logits=True,
        )
        if self.best_in_multi_mask:
            # sam_score: n
            sam_score, max_iou_idx = torch.max(sam_score, dim=1)
            # mask_pred: n,h,w
            mask_pred = mask_pred[torch.arange(mask_pred.size(0)),
                                  max_iou_idx]
        else:
            # Tensor(n,h,w), raw mask pred
            # n,1,h,w->n,h,w
            mask_pred = mask_pred.squeeze(1)
            # n,1->n
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

    # not implemented:
    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def extract_feat(self, imgs):
        raise NotImplementedError
