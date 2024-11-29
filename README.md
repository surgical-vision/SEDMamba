# SEDMamba: Enhancing Selective State Space Modelling with Bottleneck Mechanism and Fine-to-Coarse Temporal Fusion for Efficient Error Detection in Robot-Assisted Surgery
This repository provides the official PyTorch implementation of the following paper:
> [**SEDMamba: Enhancing Selective State Space Modelling with Bottleneck Mechanism and Fine-to-Coarse Temporal Fusion for Efficient Error Detection in Robot-Assisted Surgery**](https://doi.org/10.1109/LRA.2024.3505818)<br>
> [Jialang Xu](https://www.researchgate.net/profile/Jialang-Xu), Nazir Sirajudeen, Matthew Boal, Nader Francis, Danail Stoyanov, Evangelos B. Mazomenos

[2024-11-29] Release the SEDMamba code.

## Introduction
Automated detection of surgical errors can improve robotic-assisted surgery. Despite promising progress, existing methods still face challenges in capturing rich temporal context to establish long-term dependencies while maintaining computational efficiency. In this paper, we propose a novel hierarchical model named SEDMamba, which incorporates the selective state space model (SSM) into surgical error detection, facilitating efficient long sequence modelling with linear complexity. SEDMamba enhances selective SSM with a bottleneck mechanism and fine-to-coarse temporal fusion (FCTF) to detect and temporally localize surgical errors in long videos. The bottleneck mechanism compresses and restores features within their spatial dimension, thereby reducing computational complexity. FCTF utilizes multiple dilated 1D convolutional layers to merge temporal information across diverse scale ranges, accommodating errors of varying duration. Our work also contributes the first-of-its-kind, frame-level, in-vivo surgical error dataset to support error detection in real surgical cases. Specifically, we deploy the clinically validated observational clinical human reliability assessment tool (OCHRA) to annotate the errors during suturing tasks in an open-source radical prostatectomy dataset (SAR-RARP50). Experimental results demonstrate that our SEDMamba outperforms state-of-the-art methods with at least 1.82% AUC and 3.80% AP performance gains with significantly reduced computational complexity.

## Architecture
<img src="https://github.com/wzjialang/SEDMamba/blob/main/figure/framework.png"/>

Fig.1 The pipeline of the proposed SEDMamba. (a) The overall architecture of SEDMamba. (b) The fundamental block of SEDMamba, namely the bottleneck
 multi-scale state space (BMSS) block; (c) Fine-to-coarse temporal fusion (FCTF); (d) Bottleneck mechanism.

## SAR-RARP50 Dataset with Error Annotations
TBD

## Setup & Usage for the Code
### Install dependencies:
```
- mamba-ssm
- Python 3.9+
- PyTorch 1.12+
- CUDA 11.6+
```

- Simple example of dependency installation:
```
conda create -n sedmamba python=3.9
conda activate sedmamba
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mamba-ssm
pip install mamba-ssm[causal-conv1d]
pip install mamba-ssm[dev]
```

### Inference & Visualization & Train
#### Inference SEDMamba model:
```
python inference.py -exp Inference_SEDMamba -dp SEDMamba/data -cls 1 -gpu_id cuda:0 -w 4 -weight SEDMamba/weight/SEDMamba.pth
```
- `-exp`: experiment name
- `-dp`: dataset path
- `-cls`: class number (binary=1)
- `-gpu_id`: single gpu id
- `-w`: number of workers
- `-weight`: model weight path

#### Visualization results:
```
python visualization.py -lp ./exp_log/0.0001/Inference_SEDMamba -rp ./exp_log/0.0001/Inference_SEDMamba -sp ./exp_log/0.0001/Inference_SEDMamba
```
- `-lp`: ground truth path
- `-rp`: prediction result path
- `-sp`: visualization saving path

#### Train SEDMamba model:
```
python train.py -exp SEDMamba -dp SEDMamba/data -e 200 -l 1e-4 -cls 1 -gpu_id cuda:0 -w 4 -s 0
```
- `-e`: epoches
- `-l`: learning rate
- `-s`: random seed

## Acknowledge
We sincerely appreciate the authors for releasing the following valuable resources: [UVA-DSA/ExecErr_Detect_Siamese](https://github.com/UVA-DSA/ExecErr_Detect_Siamese), [TeCNO](https://github.com/tobiascz/TeCNO), [MS-TCN](https://github.com/yabufarha/ms-tcn), [ASFormer](https://github.com/ChinaYi/ASFormer), [MS-TCN++](https://github.com/sj-li/MS-TCN2), [Mamba](https://github.com/state-spaces/mamba), [Vim](https://github.com/kyegomez/VisionMamba).

## Cite
If this repository is useful for your research, please cite:
```
@ARTICLE{10766660,
  author={Xu, Jialang and Sirajudeen, Nazir and Boal, Matthew and Francis, Nader and Stoyanov, Danail and Mazomenos, Evangelos B.},
  journal={IEEE Robotics and Automation Letters}, 
  title={SEDMamba: Enhancing Selective State Space Modelling with Bottleneck Mechanism and Fine-to-Coarse Temporal Fusion for Efficient Error Detection in Robot-Assisted Surgery}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2024.3505818}}
```

