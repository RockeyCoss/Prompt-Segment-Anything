# Prompt-Segment-Anything
This is an implementation of zero-shot instance segmentation using [Segment Anything](https://github.com/facebookresearch/segment-anything). Thanks to the authors of Segment Anything for their wonderful work! 

This repository is based on [MMDetection](https://github.com/open-mmlab/mmdetection) and includes some code from [H-Deformable-DETR](https://github.com/HDETR/H-Deformable-DETR) and [FocalNet-DINO](https://github.com/FocalNet/FocalNet-DINO).

![example1](assets/example1.jpg)

## Catalog

- [x] Support Swin-L+H-Deformable-DETR+SAM
- [x] Support FocalNet-L+DINO+SAM

## Results

|         Detector         |    SAM    | Detector's Box AP | Mask AP |                            Config                            |
| :----------------------: | :-------: | :---------------: | :-----: | :----------------------------------------------------------: |
| Swin-L+H-Deformable-DETR | sam-vit-b |       58.0        |  42.5   | [config](https://github.com/RockeyCoss/Instance-Segment-Anything/blob/master/projects/configs/hdetr/swin-l-hdetr_sam-vit-b.py) |
| Swin-L+H-Deformable-DETR | sam-vit-l |       58.0        |  46.3   | [config](https://github.com/RockeyCoss/Instance-Segment-Anything/blob/master/projects/configs/hdetr/swin-l-hdetr_sam-vit-l.py) |
|     FocalNet-L+DINO      | sam-vit-b |       63.2        |  44.5   | [config](https://github.com/RockeyCoss/Instance-Segment-Anything/blob/master/projects/configs/hdetr/swin-l-hdetr_sam-vit-b.py) |

## Installation

We test the models under `python=3.7.10,pytorch=1.10.2,cuda=10.2`. Other versions might be available as well.

1. Clone this repository

```
git clone https://github.com/RockeyCoss/Instance-Segment-Anything
cd Instance-Segment-Anything
```

2. Install PyTorch

```bash
# an example
pip install torch torchvision
```

3. Install MMCV

```
pip install -U openmim
mim install "mmcv>=2.0.0"
```

4. Install MMDetection's requirements

```
pip install -r requirements.txt
```

5. Compile CUDA operators

```bash
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..
```

## Prepare COCO Dataset

Please refer to [data preparation](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html).

## Prepare Checkpoints

1. Install wget

```
pip install wget
```

2. SAM checkpoints

```bash
mkdir ckpt
cd ckpt
python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```

3. Swin-L+H-Deformable-DETR and FocalNet-L+DINO checkpoints

```bash
cd ckpt
python -m wget https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/decay0.05_drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth -o swin_l_hdetr.pth
python -m wget https://projects4jw.blob.core.windows.net/focalnet/release/detection/focalnet_large_fl4_o365_finetuned_on_coco.pth -o focalnet_l_dino.pth
cd ..

# convert checkpoints
python tools/convert_ckpt.py ckpt/swin_l_hdetr.pth ckpt/swin_l_hdetr.pth
python tools/convert_ckpt.py ckpt/focalnet_l_dino.pth ckpt/focalnet_l_dino.pth
```

## Run Evaluation

1. Evaluate Metrics

```bash
# single GPU
python tools/test.py path/to/the/config/file --eval segm
# multiple GPUs
bash tools/dist_test.sh path/to/the/config/file num_gpus --eval segm
```

2. Visualize Segmentation Results

```bash
python tools/test.py path/to/the/config/file --show-dir path/to/the/visualization/results
```
## More Segmentation Examples

![example2](assets/example2.jpg)
![example3](assets/example3.jpg)
![example4](assets/example4.jpg)
![example5](assets/example5.jpg)

## Citation

**Segment Anything**

```latex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
**H-Deformable-DETR**

```latex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}
```
**Swin Transformer**

```latex
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
**DINO**

```latex
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
**FocalNet**

```latex
@misc{yang2022focalnet,  
  author = {Yang, Jianwei and Li, Chunyuan and Dai, Xiyang and Yuan, Lu and Gao, Jianfeng},
  title = {Focal Modulation Networks},
  publisher = {arXiv},
  year = {2022},
}
```