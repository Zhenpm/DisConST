# DisConST
spatial domain identification
![image](https://github.com/Zhenpm/DisConST/blob/main/DisConST.png)

## Overview

Spatial transcriptomics (ST) is a cutting-edge technology that comprehensively characterizes gene expression patterns from a spatial perspective. Spatial domain identification, a pivotal research direction within spatial transcriptomics, serves as a critical reference for in-depth exploration of tissue organization, biological development, and disease studies, among other downstream analyses. In this work, we introduce DisConST, a novel approach that employs an optimization strategy based on zero-inflated negative binomial (ZINB) distribution and graph contrastive learning to capture the most effective representations of ST data. The representations has integrated spatial position information, transcriptome, and cell type proportions. Validation of DisConST across diverse datasets, including normal tissues and organs from various sequencing platforms, diseased tissues, and embryos, consistently yielded higher spatial domain recognition accuracy than existing methods. Meanwhile, these experiments also validated the value of DisConST in tissue organization, disease research, and biological development.

## Software dependencies

scanpy==1.9.6 
pytorch==1.12.0+cu11.3 
pytorch_geometric==2.4.0 
R==4.2.3 
mclust==5.4.10 

## set up

## Datasets
In this work, we employed five ST datasets to DisConST, including 12 slices of Human Dorsolateral Prefrontal Cortex (DLPFC), 2 slices of mouse olfactory bul tissue from Stereo-seq and ST sequencing technology, 2 sections of mouse brain sagittal slices, a human breast cancer slice and  9 slices of Mouse Organogenesis Spatiotemporal Transcriptomic Atlas (MOSTA) from Streo-seq and seqFiSH sequencing technology.
Five datasets can be downloaded from https://github.com/hannshu/st_datasets or https://pan.baidu.com/s/1mmMWKz-GaHqvjTQ-fZ1IZA?pwd=1234

## Tutorial
