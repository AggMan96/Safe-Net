# [Safe-Net]A Self-Adaptive Feature Extraction Method for Aerial-view Geo-localization

Code for Safe-Net.

## Prerequisites

- torch
- torchvision
- numpy 
- pyyaml
- tqdm
- scipy
- matplotlib
- pillow

## Dataset & Preparation
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request and put them under the `./data/` folder. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

## Pretrained Vit-S weights
You can download the pretrained Vit-S weights from the following link and put it in the **./models/pretrain_model** folder

- [Google Driver](https://drive.google.com/file/d/1gEfXVVdL827J5SJ4KGOHhdWoeDZ7UjTd/view?usp=sharing)

## Train & Evaluation
### Train & Evaluation on **University-1652**
```
bash run_train_test_U1652.sh
```
* You can change the **data_dir** and **test_dir** to your own dataset paths in **run_train_test_U1652.sh**. 

## TO-DO List

- [ ] Support SUES-200 dataset
- [ ] Support the evaluation for different levels of distance
- [ ] Support ResNet-50 backbone
- [ ] Adding the demo of FPM and FAM
- [ ] ...

## Reference
- **University-1652**: [pdf](https://arxiv.org/abs/2002.12186)|[code](https://github.com/layumi/University1652-Baseline)

- **LPN**: [pdf](https://arxiv.org/abs/2008.11646)|[code](https://github.com/wtyhub/LPN)

- **RK-Net**: [pdf](https://ieeexplore.ieee.org/document/9779991)|[code](https://github.com/AggMan96/RK-Net)

- **FSRA**: [pdf](https://arxiv.org/abs/2201.09206)|[code](https://github.com/Dmmm1997/FSRA)
