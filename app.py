import os

SPACE_ID = os.getenv('SPACE_ID')
if SPACE_ID is not None:
    # running on huggingface space
    os.system(r'mkdir ckpt')
    os.system(
        r'python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o ckpt/sam_vit_b_01ec64.pth')
    os.system(
        r'python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -o ckpt/sam_vit_l_0b3195.pth')
    os.system(
        r'python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o ckpt/sam_vit_h_4b8939.pth')

    os.system(
        r'python -m wget https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1'
        r'/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth -o ckpt/r50_hdetr.pth')
    os.system(
        r'python -m wget https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1'
        r'/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth -o ckpt/swin_t_hdetr.pth')
    os.system(
        r'python -m wget https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/decay0.05_drop_path0'
        r'.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth -o ckpt/swin_l_hdetr.pth')
    os.system(r'python -m wget https://projects4jw.blob.core.windows.net/focalnet/release/detection'
              r'/focalnet_large_fl4_o365_finetuned_on_coco.pth -o ckpt/focalnet_l_dino.pth')

    os.system(r'python tools/convert_ckpt.py ckpt/r50_hdetr.pth ckpt/r50_hdetr.pth')
    os.system(r'python tools/convert_ckpt.py ckpt/swin_t_hdetr.pth ckpt/swin_t_hdetr.pth')
    os.system(r'python tools/convert_ckpt.py ckpt/swin_l_hdetr.pth ckpt/swin_l_hdetr.pth')
    os.system(r'python tools/convert_ckpt.py ckpt/focalnet_l_dino.pth ckpt/focalnet_l_dino.pth')
import warnings
from collections import OrderedDict
from pathlib import Path

import gradio as gr
import numpy as np
import torch

import mmcv
from mmcv import Config
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE

from mmdet.core import get_classes
from mmdet.datasets import (CocoDataset, replace_ImageToTensor)
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.utils import (compat_cfg, replace_cfg_vals, setup_multi_processes,
                         update_data_root)

config_dict = OrderedDict([('r50-hdetr_sam-vit-b', 'projects/configs/hdetr/r50-hdetr_sam-vit-b.py'),
                           ('r50-hdetr_sam-vit-l', 'projects/configs/hdetr/r50-hdetr_sam-vit-l.py'),
                           ('swin-t-hdetr_sam-vit-b', 'projects/configs/hdetr/swin-t-hdetr_sam-vit-b.py'),
                           ('swin-t-hdetr_sam-vit-l', 'projects/configs/hdetr/swin-t-hdetr_sam-vit-l.py'),
                           ('swin-l-hdetr_sam-vit-b', 'projects/configs/hdetr/swin-l-hdetr_sam-vit-b.py'),
                           ('swin-l-hdetr_sam-vit-l', 'projects/configs/hdetr/swin-l-hdetr_sam-vit-l.py'),
                           # ('swin-l-hdetr_sam-vit-h', 'projects/configs/hdetr/swin-l-hdetr_sam-vit-l.py'),
                           ('focalnet-l-dino_sam-vit-b', 'projects/configs/focalnet_dino/focalnet-l-dino_sam-vit-b.py'),
                           # ('focalnet-l-dino_sam-vit-l', 'projects/configs/focalnet_dino/focalnet-l-dino_sam-vit-l.py'),
                           # ('focalnet-l-dino_sam-vit-h', 'projects/configs/focalnet_dino/focalnet-l-dino_sam-vit-h.py')
                           ])


def init_demo_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.
    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.
    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif (config.model.get('backbone', None) is not None
          and 'init_cfg' in config.model.backbone):
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model)
        model.cfg = config

    return model


def inference_demo_detector(model, imgs):
    """Inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    ori_img = imgs
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data, ori_img=ori_img)

    if not is_batch:
        return results[0]
    else:
        return results


def inference(img, config):
    if img is None:
        return None
    print(f"config: {config}")
    config = config_dict[config]
    cfg = Config.fromfile(config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if IS_CUDA_AVAILABLE or IS_MLU_AVAILABLE:
        device = "cuda"
    else:
        device = "cpu"
    model = init_demo_detector(cfg, None, device=device)
    model.CLASSES = CocoDataset.CLASSES

    results = inference_demo_detector(model, img)
    visualize = model.show_result(
        img,
        results,
        bbox_color=CocoDataset.PALETTE,
        text_color=CocoDataset.PALETTE,
        mask_color=CocoDataset.PALETTE,
        show=False,
        out_file=None,
        score_thr=0.3
    )
    del model
    return visualize


description = """
#  <center>Prompt Segment Anything (zero-shot instance segmentation demo)</center>
Github link: [Link](https://github.com/RockeyCoss/Prompt-Segment-Anything)
You can select the model you want to use from the "Model" dropdown menu and click "Submit" to segment the image you uploaded to the "Input Image" box.
"""
if SPACE_ID is not None:
    description += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'


def main():
    with gr.Blocks() as demo:
        gr.Markdown(description)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="numpy", label="Input Image")
                    model_type = gr.Dropdown(choices=list(config_dict.keys()),
                                             value=list(config_dict.keys())[0],
                                             label='Model',
                                             multiselect=False)
                    with gr.Row():
                        clear_btn = gr.Button(value="Clear")
                        submit_btn = gr.Button(value="Submit")
                output_img = gr.Image(type="numpy", label="Output")
            gr.Examples(
                examples=[["./assets/img1.jpg", "r50-hdetr_sam-vit-b"],
                          ["./assets/img2.jpg", "r50-hdetr_sam-vit-b"],
                          ["./assets/img3.jpg", "r50-hdetr_sam-vit-b"],
                          ["./assets/img4.jpg", "r50-hdetr_sam-vit-b"]],
                inputs=[input_img, model_type],
                outputs=output_img,
                fn=inference
            )

        submit_btn.click(inference,
                         inputs=[input_img, model_type],
                         outputs=output_img)
        clear_btn.click(lambda: [None, None], None, [input_img, output_img], queue=False)

    demo.queue()
    demo.launch()


if __name__ == '__main__':
    main()
