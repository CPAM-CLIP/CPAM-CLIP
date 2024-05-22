# CPAM-CLIP: Training-Free Dense Localization in Vision-Language Model via Class Probability Guided Attention Map

**Official PyTorch implementation of CPAM-CLIP**


## Dependencies

This repo is built on top of [CLIP](https://github.com/openai/CLIP), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [SCLIP](https://github.com/wangf3014/SCLIP/blob/main/README.md). To run CPAM-CLIP, please install the following packages with your Pytorch environment. We recommend using Pytorch==1.9.x for better compatibility to the following MMSeg version.

```
conda create -n cpamclip python=3.9
conda activate cpamclip
pip install -r requirements.txt 
```

## Datasets
We include the following dataset configurations in this repo: PASCAL VOC, PASCAL Context and COCO-Object.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

**Remember to modify the dataset paths in the config files in** `config/cfg_DATASET.py`



## Run CPAM-CLIP

```
python eval.py --config ./configs/cfg_DATASET.py --workdir YOUR_WORK_DIR
```