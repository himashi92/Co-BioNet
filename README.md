# Co-BioNet: Uncertainty-Guided Dual-Views for Semi-Supervised Volumetric  Medical Image Segmentation
This repo contains the supported pytorch code and configuration files to reproduce results of Uncertainty-Guided Dual-Views for Semi-Supervised Volumetric Medical Image Segmentation Article.

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

Pre-processed data can be found in folder data.

## Running Demo
Demonstration is created on generating segmentation masks on a sample of unseen Pancreas CT with trained torch models on 10% and 20% Labeled Pancreas CT and Left Atrial MRI data. You can run the given python notebook in the demo folder.

## Train Model
### To train the model for Pancreas CT dataset on 10% Lableled data
```bash
cd code
nohup python train_cobionet_PANCREAS.py --labelnum 6 --lamda 1.0 --consistency 1.0 --mu 0.01 --t_m 0.2 --max_iteration 15000 &> pa_10_perc.out &
```

### To train the model for Pancreas CT dataset on 20% Lableled data
```bash
cd code
nohup python train_cobionet_PANCREAS.py --labelnum 12 --lamda 1.0 --consistency 1.0 --mu 0.01 --t_m 0.2 --max_iteration 15000 &> pa_20_perc.out &
```

### To train the model for Left Atrial MRI dataset on 10% Lableled data
```bash
cd code
nohup python train_cobionet_LA.py --labelnum 8 --lamda 0.8 --consistency 1.0 --mu 0.01 --t_m 0.3 --max_iteration 15000 &> la_10_perc.out &
```

### To train the model for Left Atrial MRI dataset on 20% Lableled data
```bash
cd code
nohup python train_cobionet_LA.py --labelnum 16 --lamda 0.8 --consistency 1.0 --mu 0.01 --t_m 0.3 --max_iteration 15000 &> la_20_perc.out &
```

It would take around 4 hours to complete model training.

## Test Model

### To test the model 1 for Pancreas CT dataset on 10% Lableled data
```bash
cd code
python eval_3d.py --dataset_name Pancreas_CT --labelnum 6 --model_num 1
```

### To test the ensemble model for Pancreas CT dataset on 10% Lableled data
```bash
cd code
python eval_3d_ensemble.py --dataset_name Pancreas_CT --labelnum 6
```

### To test and get best segmentation masks that are more closer to ground truth annotations out of model 1, model 2 and the ensemble model for Pancreas CT dataset on 10% Lableled data
```bash
cd code
python eval_get_best.py --dataset_name Pancreas_CT --labelnum 6
```

## Acknowledgements

This repository makes liberal use of code from [SASSNet](https://github.com/kleinzcy/SASSnet), [UAMT](https://github.com/yulequan/UA-MT), [DTC](https://github.com/HiLab-git/DTC) and [MC-Net](https://github.com/ycwu1997/MC-Net/)

