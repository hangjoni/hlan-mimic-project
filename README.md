#### Download pre-trained embeddings and processed datasets to ./embeddings and ./datasets
```
curl https://bd4hstorage.blob.core.windows.net/bd4h/datasets.zip --output datasets.zip && unzip datasets.zip && rm datasets.zip
curl https://bd4hstorage.blob.core.windows.net/bd4h/embeddings.zip --output embeddings.zip && unzip datasets.zip && rm embeddings.zip
```
#### Install dependencies
```
pip install -r requirements.txt
```

#### Run training
```
python ./HLAN_pytorch/train.py --epochs 1 --lr 0.0005
```
for debugging purpose, it is useful to run training with just a few data points, see HLAN_pytorch/train_playground.ipynb for example
