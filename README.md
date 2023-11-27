# Pytorch implementation for Hierarchical Label-wise Attention Network ("HLAN")

This model was described in [this paper](https://arxiv.org/abs/2010.15728). The author shared an implementation in this repo that rely on Python 3.7 and Tensorflow 1.

In this project, we reproduced the claim made by the paper. We also re-implemented the described HLAN architecture using Pytorch

## Steps to run training

### Download pre-trained embeddings and processed datasets to ./embeddings and ./datasets

```
curl https://bd4hstorage.blob.core.windows.net/bd4h/datasets.zip --output datasets.zip && unzip datasets.zip && rm datasets.zip
curl https://bd4hstorage.blob.core.windows.net/bd4h/embeddings.zip --output embeddings.zip && unzip datasets.zip && rm embeddings.zip
```

The folder structure should be something like this:

```
hlan-mimic-project/
   cache_vocabulary_label_pik/
   datasets/
   embeddings/
   HLAN_pytorch/
```

### Install dependencies

```
conda create -n hlan python=3.11
conda activate hlan
pip install -r requirements.txt
```

### Training

Make sure to be inside HLAN_pytorch folder when triggering training.

#### Run training with mini dataset

mini dataset has 10 data points. Use this option when testing the training script as it is fast to see training complete.

```
cd HLAN_pytorch
python train.py --epochs 5 --mini
```

#### Run training and log to Weights and Biases

```
cd HLAN_pytorch
python train.py --epochs 5 --log --verbose
```

This requires a Weights and Biases account. Follow the steps [here](https://docs.wandb.ai/quickstart) to create an account and login using Terminal.

Training loss and metrics will be logged to Weights and Biases when setting `--log`

use `--verbose` if want to print every epoch results to the console

#### Resume training with a previously saved checkpoint

By default, we save the best and last checkpoint to ./checkpoints folder during training. To resume training with that checkpoint, pass the folder name to train.py

```
cd HLAN_pytorch
python train.py --epochs 5 --log --checkpoint_to_resume_from ../checkpoints/20231127_1604_qxDMJ/last.pt
```

#### Train with GPU or MPS

To train with GPU, use `--device gpu`
To train with Mac GPU MPS, use `--device mps`

```
cd HLAN_pytorch
python train.py --epochs 5 --device mps --mini
```

#### Train with or without Label Embedding Initalization

To train Label Embedding initialization `--le`

```
cd HLAN_pytorch
python train.py --epochs 5 --mini --le
```

To train without, simply omit .
