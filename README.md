# Co-BioNet: Uncertainty-Guided Dual-Views for Semi-Supervised Volumetric  Medical Image Segmentation
This repo contains the supported pytorch code and configuration files to reproduce results of Uncertainty-Guided Dual-Views for Semi-Supervised Volumetric Medical Image Segmentation Article.

## Abstract

Deep learning has resulted in tremendous progress in the field of medical AI. However,  training deep learning models usually require large amounts of annotated data. Annotating large-scale datasets, is prone to human biases and often very laborious, especially for dense prediction tasks such as image segmentation. Inspired by semi-supervised algorithms that employ both labeled and unlabeled data for training, we propose a  dual-view framework based on adversarial learning for segmenting volumetric images. In doing so, we employ critic networks to allow each view to learn from high-confidence predictions of the other view via measuring a notion of uncertainty. Furthermore, to jointly learn the dual-views and the critics, we formulate the learning problem as a min-max problem. We analyze and contrast our proposed method against state-of-the-art baselines, both qualitatively and quantitatively, on four public datasets with multiple modalities (\eg, CT and MRI) and demonstrate that the proposed semi-supervised method substantially outperforms the competing baselines while achieving competitive performance compared to fully-supervised counterparts. Our empirical results suggest that an uncertainty-guided co-training framework can make two neural networks robust to data artifacts and have the ability to generate plausible segmentation masks that can be helpful for semi-automated segmentation processes.  

## Link to full paper:
To be Added

## Proposed Architecture
![Proposed Architecture](img/co_bionet_architecture.png?raw=true)

## System requirements
Under this section, we provide details on environmental setup and dependencies required to train/test the Co-BioNet model.
This software was originally designed and run on a system running Ubuntu (Compatible with Windows 11 as well).
<br>
All the experiments are conducted on Ubuntu 20.04 Focal version with Python 3.8.
<br>
To train Co-BioNet with the given settings, the system requires a GPU with at least 24GB. All the experiments are conducted on Nvidia RTX 3090 single GPU.
(Not required any non-standard hardware)
<br>
To test model's performance on unseen test data, the system requires a GPU with at least 4 GB.

### Create a virtual environment

```bash 
pip install virtualenv
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
```

### Installation guide 

- Install torch : 
```bash
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
- Install other dependencies :
```bash 
pip install -r requirements.txt
```

### Typical Install Time 
This depends on the internet connection speed. It would take around 15-30 minutes to create environment and install all the dependencies required.

## Dataset Preparation
The experiments are conducted on two publicly available datasets,
- National Institutes of Health (NIH) Panceas CT Dataset : https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
- 2018 Left Atrial Segmentation Challenge Dataset : http://atriaseg2018.cardiacatlas.org
- MSD BraTS Dataset : http://medicaldecathlon.com/

Pre-processed data can be found in folder data.

## Figshare Project Page
All the pre-trained models, figures, evaluations, a video on how the training pipeline works, and the source code are included in this project page [link](https://figshare.com/projects/Uncertainty-Guided_Dual-Views_for_Semi-Supervised_Volumetric_Medical_Image_Segmentation/158963)  

- DOI : https://doi.org/10.6084/m9.figshare.22140194.v5

## Trained Model Weights
Download trained model weights from this shared drive [link](https://drive.google.com/drive/folders/1O8GmlquR2ZS6-PBTBp9d4GSWg06Z-uwa?usp=sharing), and put it under folder **code/model** or **code_msd_brats/model**

## Running Demo
Demonstration is created on generating segmentation masks on a sample of unseen Pancreas CT with trained torch models on 10% and 20% Labeled Pancreas CT and Left Atrial MRI data. You can run the given python notebook in the demo folder.

## Train Model
- To train the model for Pancreas CT dataset on 10% Lableled data
```bash
cd code
nohup python train_cobionet_semi.py --dataset_name Pancreas_CT --labelnum 6 --lamda 1.0 --consistency 1.0 --mu 0.01 --t_m 0.2 --max_iteration 15000 &> pa_10_perc.out &
```

- To train the model for Pancreas CT dataset on 20% Lableled data
```bash
cd code
nohup python train_cobionet_semi.py --dataset_name Pancreas_CT --labelnum 12 --lamda 1.0 --consistency 1.0 --mu 0.01 --t_m 0.2 --max_iteration 15000 &> pa_20_perc.out &
```

- To train the model for Left Atrial MRI dataset on 10% Lableled data
```bash
cd code
nohup python train_cobionet_semi.py --dataset_name LA --labelnum 8 --lamda 0.7 --consistency 1.0 --mu 0.01 --t_m 0.4 --max_iteration 15000 &> la_10_perc.out &
```

- To train the model for Left Atrial MRI dataset on 20% Lableled data
```bash
cd code
nohup python train_cobionet_semi.py --dataset_name LA --labelnum 16 --lamda 0.7 --consistency 1.0 --mu 0.01 --t_m 0.4 --max_iteration 15000 &> la_20_perc.out &
```

It would take around 5 hours to complete model training. You can try out different hyper-parameter settings and further improve the accuracy.

## Test Model

- To test the Co-BioNet ensemble model for Pancreas CT dataset on 10% Lableled data
```bash
cd code
python eval_3d_ensemble.py --dataset_name Pancreas_CT --labelnum 6
```

## Acknowledgements

This repository makes liberal use of code from [SASSNet](https://github.com/kleinzcy/SASSnet), [UAMT](https://github.com/yulequan/UA-MT), [DTC](https://github.com/HiLab-git/DTC) and [MC-Net](https://github.com/ycwu1997/MC-Net/)

## Citing Co-BioNet

If you find this repository useful, please consider giving us a star ⭐ and cite our work:

```bash
      Peiris, Himashi (2023): Project Contributions. figshare. Journal contribution. https://doi.org/10.6084/m9.figshare.22140194.v5
    }
```

