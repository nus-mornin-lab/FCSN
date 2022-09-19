# [FCSN: Global Context Aware Segmentation by Learning the Fourier Coefficients of Objects in Medical Images](https://arxiv.org/abs/2207.14477)

### Authors
*  Jeon Young Seok (co-first) (youngseokejeon74@gmail.com)
*  Hongfei Yang (co-first)  (hfyang@nus.edu.sg)
*  Mengling Feng (coressponding) (ephfm@gmail.com)

---
## Abstract
>The encoder-decoder model is a commonly used Deep Neural Network (DNN) model for medical image segmentation. Conventional encoder-decoder models make pixel-wise predictions focusing heavily on local patterns around the pixel. This makes it challenging to give segmentation that preserves the object’s shape and topology, which often requires an  understanding of the global context of the object. In this work, we propose a Fourier Coefficient Segmentation Network (FCSN)—a novel DNNbased model that segments an object by learning the complex Fourier coefficients of the object’s masks. The Fourier coefficients are calculated by integrating over the whole contour. Therefore, for our model to make a precise estimation of the coefficients, the model is motivated to incorporate the global context of the object, leading to a more accurate segmentation of the object’s shape. This global context awareness also makes our model robust to unseen local perturbations during inference, such as additive noise or motion blur that are prevalent in medical images. When FCSN is compared with other state-of-the-art models (UNet+, DeepLabV3+, UNETR) on 3 medical image segmentation tasks (ISIC_2018, RIM_CUP, RIM_DISC), FCSN attains significantly lower Hausdorff scores of 19.14 (6%), 17.42 (6%), and 9.16 (14%) on the 3 tasks, respectively. Moreover, FCSN is lightweight by discarding the decoder module, which incurs significant computational overhead. FCSN only requires 22.2M parameters, 82M and 10M fewer parameters than UNETR and DeepLabV3+. FCSN attains inference and training speeds of 1.6ms/img and 6.3ms/img, that is 8× and 3× faster than UNet and UNETR.

### Keywords 
* Medical Image Segmentation
* Global Context Aware Learning
* Decoder-Free Segmentation
---
## To Run
###  1. Data

We test FCSN on 2 medical image datasets: 1.[ISIC_2018]() and 2: [RIM_ONE_DL](). Both datasets are publicly available. To download, follow the instructions below.

####  ISIC_2018
* ISIC_2018 data can be downloaded [here](). Donwload only the train-sets which are highlighted with red boxes in the figure below. We use 5-fold cross-validation on the training set for model evaluation because the ground-truth for the test-set is not publictly available.

![](./imgs/data_ISIC.png)

* The originial dataset has varying image size. It is required for users to resize all the images to 256 x 256 before training.

* Once the resizing is done, store the processed images and its corressponding masks in **project/data/datasets/** directory as shown below.
```
ISIC
└── processed
    ├── train
    │   ├── images [2594  jpgs]
    │   └── masks [2594  jpgs]
```
####  RIM_ONE_DL
* RIM_ONE_DL data can be downloaded [here](). Donwload both the train-set and test-set clicking the link hightlighted with a red box.

![](./imgs/data_RIM.png)

* It is required for users to resize all the images to 256 x 256 before training.

* You are required to separate the two tasks (RIM_DISC and RIM_CUP) as independant datasets.

* Once the resizing and the separation of tasks are done, store the processed images and its corressponding masks in **project/data/datasets/** directory as shown below
```
RIM_CUP
└── processed
    ├── RIM-ONE_DL_256_reference_segmentations
    │   ├── normal [626 entries exceeds filelimit, not opening dir]
    │   └── glaucoma [344 entries exceeds filelimit, not opening dir]
    └── RIM-ONE_DL_256_images
        ├── partitioned_randomly
        │   ├── training_set
        │   │   ├── normal [219 entries exceeds filelimit, not opening dir]
        │   │   └── glaucoma [120 entries exceeds filelimit, not opening dir]
        │   └── test_set
        │       ├── normal [94 entries exceeds filelimit, not opening dir]
        │       └── glaucoma [52 entries exceeds filelimit, not opening dir]
```

```
RIM_DISC
└── processed
    ├── RIM-ONE_DL_256_reference_segmentations
    │   ├── normal [626 entries exceeds filelimit, not opening dir]
    │   └── glaucoma [344 entries exceeds filelimit, not opening dir]
    └── RIM-ONE_DL_256_images
        ├── partitioned_randomly
        │   ├── training_set
        │   │   ├── normal [219 entries exceeds filelimit, not opening dir]
        │   │   └── glaucoma [120 entries exceeds filelimit, not opening dir]
        │   └── test_set
        │       ├── normal [94 entries exceeds filelimit, not opening dir]
        │       └── glaucoma [52 entries exceeds filelimit, not opening dir]
````

### 2. Train
To train FCSN with the default setting as given in the paper run **./train.sh** with appropreate arguments.
For example, if you want to run the default FCSN setting for ISIC task, the command is
```sh
./train.sh ISIC FFTDSNTNet ignore ResNet 21 0
```
To run baseline model such as UNet, the command is
```sh
./train.sh ISIC UNet standard ignore 21 0
```

### 3. Evaluation 

We evaluate our model bloadly on 4 aspects: 1) performance without noise, 2) performance with noise, 3) model complexity, and 4) receptive field size. We provide jupyter-notebook in **evaluation\** directory.

* 1) performance without noise : VisualizeResults
* 2) performance with noise : VisualizeNoise
* 3) model complexity : VisualizeComplexity
* 4) receptive field size : VisualizeReceptive
