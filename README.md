# DisConST
![image](https://github.com/Zhenpm/DisConST/blob/main/DisConST.png)

## Software dependencies

scanpy==1.9.6 <br />
pytorch==1.12.0+cu11.3 <br />
pytorch_geometric==2.4.0 <br />
R==4.2.3 <br />
mclust==5.4.10 <br />

## set up

First clone the repository. 
```
git clone https://github.com/Zhenpm/DisConST.git 
cd DisConST-main
```
Then, we suggest creating a new environment： <br />
```
conda create -n disconst python=3.10 
conda activate disconst
```
Additionally, install the packages required: <br />
```
pip install -r requiements.txt
``` 

## Datasets
In this work, we employed five ST datasets to DisConST, including 12 slices of Human Dorsolateral Prefrontal Cortex (DLPFC), 2 slices of mouse olfactory bul tissue from Stereo-seq and ST sequencing technology, 2 sections of mouse brain sagittal slices, a human breast cancer slice and  9 slices of Mouse Organogenesis Spatiotemporal Transcriptomic Atlas (MOSTA) from Streo-seq and seqFiSH sequencing technology. <br />
Five datasets can be downloaded from https://github.com/hannshu/st_datasets or https://pan.baidu.com/s/1mmMWKz-GaHqvjTQ-fZ1IZA?pwd=1234
