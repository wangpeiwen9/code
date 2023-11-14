# PraNet: Parallel Reverse Attention Network for Polyp Segmentation 

## 1 State-of-the-art Approaches  
1. "Selective feature aggregation network with area-boundary constraints for polyp segmentation." IEEE Transactions on Medical Imaging, 2019.
doi: https://link.springer.com/chapter/10.1007/978-3-030-32239-7_34 
2. "PraNet: Parallel Reverse Attention Network for Polyp Segmentation" IEEE Transactions on Medical Imaging, 2020.
doi: https://link.springer.com/chapter/10.1007%2F978-3-030-59725-2_26
3. "Hardnet-mseg: A simple encoder-decoder polyp segmentation neural network that achieves over 0.9 mean dice and 86 fps" arXiv, 2021
doi: https://arxiv.org/pdf/2101.07172.pdf
4. "TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation" arXiv, 2021.
doi: https://arxiv.org/pdf/2102.08005.pdf


## 2. Overview

### 2.1. Introduction

Colonoscopy is an effective technique for detecting colorectal polyps, which are highly related to colorectal cancer. 
In clinical practice, segmenting polyps from colonoscopy images is of great importance since it provides valuable 
information for diagnosis and surgery. However, accurate polyp segmentation is a challenging task, for two major reasons:
(i) the same type of polyps has a diversity of size, color and texture; and
(ii) the boundary between a polyp and its surrounding mucosa is not sharp. 

To address these challenges, we propose a parallel reverse attention network (PraNet) for accurate polyp segmentation in colonoscopy
images. Specifically, we first aggregate the features in high-level layers using a parallel partial decoder (PPD). 
Based on the combined feature, we then generate a global map as the initial guidance area for the following components. 
In addition, we mine the boundary cues using a reverse attention (RA) module, which is able to establish the relationship between
areas and boundary cues. Thanks to the recurrent cooperation mechanism between areas and boundaries, 
our PraNet is capable of calibrating any misaligned predictions, improving the segmentation accuracy. 

Quantitative and qualitative evaluations on five challenging datasets across six
metrics show that our PraNet improves the segmentation accuracy significantly, and presents a number of advantages in terms of generalizability,
and real-time segmentation efficiency (∼50fps).

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/framework-final-min.png"/> <br />
    <em> 
    Figure 1: Overview of the proposed PraNet, which consists of three reverse attention 
    modules with a parallel partial decoder connection. See § 2 in the paper for details.
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    <img src="./imgs/qualitative_results.png"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX TITAN GPU of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that PraNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n PraNet python=3.6`.
    + Installing necessary packages: PyTorch 1.1
    
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `train.py`.

    + Just enjoy it!

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!