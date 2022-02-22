# Spectrum Correction: Acoustic Scene Classification with Mismatched Recording Devices
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spectrum-correction-acoustic-scene/acoustic-scene-classification-on-dcase-2019)](https://paperswithcode.com/sota/acoustic-scene-classification-on-dcase-2019?p=spectrum-correction-acoustic-scene)

This repository contains source code of the experiments presented in 

["Spectrum Correction: Acoustic Scene Classification with Mismatched Recording Devices"](https://www.doi.org/10.21437/Interspeech.2020-3088).

DOI: 10.21437/Interspeech.2020-3088

![Spectrum Correction Interspeech 2020 slides thumbnail.](Interspeech%202020%20thumbnail.png)
Slides from Interspeech 2020 can be found in `./Interspeech 2020 slides.mp4`

## Table of Contents
1. [Description](#Description)
    * [Selected Results](#Results)
2. [Training](#Training)
    * [Installation](#Installation)
    * [Quick Start](#Start)
    * [Preprocessing](#Preprocessing)
    * [Experiments](#Experiments)
3. [License](#License)
4. [Citation](#Citation)

## <a name="Description"></a> Description

Machine learning algorithms, 
when trained on audio recordings from a limited set of devices, 
may not generalize well to samples recorded using other devices 
with different frequency responses. 

In this work, a relatively straightforward method is introduced 
to address this problem. 
Two variants of the approach are presented. 
First requires aligned examples from multiple devices, 
the second approach alleviates this requirement. 

This method works for both time and frequency domain representations
 of audio recordings. Further, a relation to standardization 
 and Cepstral Mean Subtraction is analysed. 
 The proposed approach becomes effective even when very few examples 
 are provided. 
 
 This method was developed during 
 the Detection and Classification of Acoustic Scenes and Events 
 (DCASE) 2019 challenge and **won the 1st place** 
 in the scenario with mismatched recording devices 
 with the accuracy of 75%. 
 
### <a name="Results"></a> Selected Results 

Accuracy for devices A, B and C for models trained
with and without *Spectrum Correction* and using other methods 
on the *TAU Urban Acoustic Scenes 2019 Mobile* dataset.
See the paper for the full table.

method| device A | device B | device C | mean
---:|:---:|:---:|:---:|:---:
*RASTA* | 60.2% | 59.7% | 60.9% | 60.1%
*PCEN* | 64.0% | 58.6% | 64.4% | 62.3%
*-* | 71.6% | 59.2% | 61.2% | 64.0%
*CMS* | 63.9% | 61.4% | 67.1% | 64.2%
*Spectrum Correction* | **73.1%** | **65.9%** | **71.4%** | **70.4%**

## <a name="Training"></a> Training

### <a name="Installation"></a> Installation

Installation requires CUDA 10.0 and the corresponding CuDNN.
The dataset will be downloaded automatically (default location is `./data/dcase/TAU-urban-acoustic-scenes-2019-mobile-development`).

```shell script
git clone <repository-url> spectrum-correction
cd spectrum-correction
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
```

### <a name="Start"></a> Quick Start

Download and preprocess the dataset, then train the model:
```shell script
. venv/bin/activate
./prepare-dcase.py 0 data/dcase.h5
./train.py --reproducible --mixup-exp --mixup 0.4 data/dcase.h5 baseline
```
The raw dataset should get downloaded and extracted automatically, when `./prepare-dcase.py` is executed for the first time.

Launch TensorBoard to view the results:
```shell script
. venv/bin/activate
tensorboard --logdir logs
```

### <a name="Preprocessing"></a> Preprocessing

Dataset for training is prepared using the `./prepare-dcase.py` script, 
which downloads and prepares the data. Options can be view using 

```shell script
./prepare-dcase.py --help
```

Dataset with default settings (0 removes the limit on the number of examples used to fit the correction):
```shell script
./prepare-dcase.py 0 data/dcase.h5
```
Dataset with default settings and maximum of 32 examples for SC:
```shell script
./prepare-dcase.py 32 data/dcase-32-pairs.h5
```
Dataset made using the aligned variant of SC and with device *b* as the reference device:
```shell script
./prepare-dcase.py --aligned --reference b 0 data/dcase-b-reference.h5
```
Dataset created using Spectrum Correction with implementation 
based on Finite Input Response (FIR) filter: 
```shell script
./prepare-dcase.py --aligned --fir 0 data/dcase-fir.h5
```
Dataset using a randomized cross validation split (seed 16)
and reusing examples from developement set: 
```shell script
./prepare-dcase.py --reuse --split 16 0 data/dcase-validation-16.h5
```

### <a name="Experiments"></a> Experiments

Training is performed using the `./train.py` script. 
Options can be viewed using
```shell script
./train.py --help
```
All experiments used the following invocation 
with different dataset and experiment name.
```shell script
./train.py --reproducible --mixup-exp --mixup 0.4 <dataset> <experiment-name>
```
For example:
```shell script
./train.py --reproducible --mixup-exp --mixup 0.4 data/dcase.h5 baseline
```

## <a name="License"></a> License

This source code is relased under *AGPL v3* license. See the LICENSE file.

## <a name="Citation"></a> Citation
If you find this repository or the paper useful, please cite them as:
```text
@inproceedings{Kosmider2020,
  author={Michał Kośmider},
  title={{Spectrum Correction: Acoustic Scene Classification with Mismatched Recording Devices}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={4641--4645},
  doi={10.21437/Interspeech.2020-3088},
  url={http://dx.doi.org/10.21437/Interspeech.2020-3088}
}
``` 
