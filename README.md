# Pytorch implementation for Hierarchical Label-wise Attention Network ("HLAN")

This model was described in . The author shared an implementation in this repo that rely on Python 3.7 and Tensorflow 1.

The Hierarchical Label-wise Attention Network was introduced in this [this paper](https://arxiv.org/abs/2010.15728). The implementation provided by the author of the paper relies on old version of Tensorflow (v1) and is 3000 lines of code. In this project, I implemented the described HLAN architecture using Pytorch, improving readability and brevity of the code (90% shorter code base). I also applied this model on patient discharge notes and predict the ICD labels (name of diagnoses), achieving the same accuracy as reported in the paper. 
Other improvements include:
- Improved logging using Weights and Biases wandb library
- Improved interpretability by using the general method provided by captum library

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

### Model interpretability

The paper argued that HLAN's major advantage over other architecture is it allows to attribute the prediction of each label to particular word in the discharge notes. We have attempted to interpret label prediction decision of the model using captum python package. See `HLAN_pytorch/visualization_playground.ipynb` for the code and outputs
