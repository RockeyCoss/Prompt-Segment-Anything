import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule

from .models import build_model
from .models.util.misc import NestedTensor, inverse_sigmoid


class HDetrWrapper(BaseModule):
    def __init__(self,
                 args=None,
                 init_cfg=None):
        super(HDetrWrapper, self).__init__(init_cfg)
        model, box_postprocessor = build_model(args)
        self.model = model
        self.box_postprocessor = box_postprocessor

        self.model.num_queries = self.model.num_queries_one2one
        self.model.transformer.two_stage_num_proposals = self.model.num_queries
        self.cls_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 84, 85, 86, 87, 88, 89, 90]

    def forward(self,
                img,
                img_metas):
        """Forward function for training mode.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        """
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        batch_size = img.size(0)
        img_masks = img.new_ones((batch_size, input_img_h, input_img_w),
                                 dtype=torch.bool)
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            img_masks[img_id, :img_h, :img_w] = False
        samples = NestedTensor(tensors=img, mask=img_masks)
        features, pos = self.model.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.model.num_feature_levels):
                if l == _len_srcs:
                    src = self.model.input_proj[l](features[-1].tensors)
                else:
                    src = self.model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.model.two_stage or self.model.mixed_selection:
            query_embeds = self.model.query_embed.weight[0: self.model.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([self.model.num_queries, self.model.num_queries, ]).bool().to(src.device)
        )
        self_attn_mask[self.model.num_queries_one2one:, 0: self.model.num_queries_one2one, ] = True
        self_attn_mask[0: self.model.num_queries_one2one, self.model.num_queries_one2one:, ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.model.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.model.class_embed[lvl](hs[lvl])
            tmp = self.model.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(
                outputs_class[:, 0: self.model.num_queries_one2one]
            )
            outputs_classes_one2many.append(
                outputs_class[:, self.model.num_queries_one2one:]
            )
            outputs_coords_one2one.append(
                outputs_coord[:, 0: self.model.num_queries_one2one]
            )
            outputs_coords_one2many.append(outputs_coord[:, self.model.num_queries_one2one:])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        sampled_logits = outputs_classes_one2one[-1][:, :, self.cls_index]
        out = {
            "pred_logits": sampled_logits,
            "pred_boxes": outputs_coords_one2one[-1],
        }
        return out

    def simple_test(self, img, img_metas, rescale=False):
        # out: dict
        out = self(img, img_metas)
        if rescale:
            ori_target_sizes = [meta_info['ori_shape'][:2] for meta_info in img_metas]
        else:
            ori_target_sizes = [meta_info['img_shape'][:2] for meta_info in img_metas]
        ori_target_sizes = out['pred_logits'].new_tensor(ori_target_sizes, dtype=torch.int64)
        # results: List[dict(scores, labels, boxes)]
        results = self.box_postprocessor(out, ori_target_sizes)
        return results
