# DisConST
spatial domain identification with DisConST
![image](https://github.com/Zhenpm/DisConST/blob/main/DisConST.png)

## Overview

Spatial transcriptomics (ST) is a cutting-edge technology that comprehensively characterizes gene expression patterns from a spatial perspective. Spatial domain identification, a pivotal research direction within spatial transcriptomics, serves as a critical reference for in-depth exploration of tissue organization, biological development, and disease studies, among other downstream analyses. In this work, we introduce DisConST, a novel approach that employs an optimization strategy based on zero-inflated negative binomial (ZINB) distribution and graph contrastive learning to capture the most effective representations of ST data. The representations has integrated spatial position information, transcriptome, and cell type proportions. Validation of DisConST across diverse datasets, including normal tissues and organs from various sequencing platforms, diseased tissues, and embryos, consistently yielded higher spatial domain recognition accuracy than existing methods. Meanwhile, these experiments also validated the value of DisConST in tissue organization, disease research, and biological development.

## Software dependencies

scanpy==1.9.6 <br />
pytorch==1.12.0+cu11.3 <br />
pytorch_geometric==2.4.0 <br />
R==4.2.3 <br />
mclust==5.4.10 <br />

## set up

First clone the repository.
`git clone https://github.com/Zhenpm/DisConST.git
cd DisConST-main`
Then, we suggest creating a new environment：
`conda create -n disconst python=3.10
conda activate disconst`
Additionally, install the packages required:
`pip install -r requiements.txt`

## Datasets
In this work, we employed five ST datasets to DisConST, including 12 slices of Human Dorsolateral Prefrontal Cortex (DLPFC), 2 slices of mouse olfactory bul tissue from Stereo-seq and ST sequencing technology, 2 sections of mouse brain sagittal slices, a human breast cancer slice and  9 slices of Mouse Organogenesis Spatiotemporal Transcriptomic Atlas (MOSTA) from Streo-seq and seqFiSH sequencing technology. <br />
Five datasets can be downloaded from https://github.com/hannshu/st_datasets or https://pan.baidu.com/s/1mmMWKz-GaHqvjTQ-fZ1IZA?pwd=1234
