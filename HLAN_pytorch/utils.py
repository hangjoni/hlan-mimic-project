import sys
sys.path.append("..")

import codecs
import random
import numpy as np
from gensim.models import Word2Vec
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from constants import *

from HLAN.data_util_gensim import create_vocabulary_label_pre_split, create_vocabulary
from torch.utils.data import DataLoader, Subset
from metrics import all_micro

def load_data_multilabel_pre_split(vocabulary_word2index, vocabulary_word2index_label, data_path='', keep_label_percent=1.0):
    print("load_data.started...")
    print("load_data_multilabel_new.data_path:", data_path)

    with codecs.open(data_path, 'r', 'latin-1') as f:
        lines = f.readlines()

    X = []
    Y = []

    for i, line in enumerate(lines):
        x, y = line.split('__label__')
        y = y.strip().replace('\n', '')
        x = x.strip()

        # Transform words to indices
        x = [vocabulary_word2index.get(word, 0) for word in x.split(" ")]
        X.append(x)

        # Transform labels to multi-hot
        labels = y.split(" ")
        label_indices = [vocabulary_word2index_label.get(label) for label in labels if vocabulary_word2index_label.get(label) is not None]
        if label_indices:
            # Truncate labels
            random.shuffle(label_indices)
            label_indices = label_indices[:round(len(label_indices) * keep_label_percent)]
            multi_hot_labels = [1 if i in label_indices else 0 for i in range(len(vocabulary_word2index_label))]
            Y.append(multi_hot_labels)

    print("load_data.ended...")
    return X, Y

def pad_or_truncate_sequence(sequence, maxlen):
    if len(sequence) > maxlen:
        sequence = sequence[:maxlen]
    elif len(sequence) < maxlen:
        sequence += [0] * (maxlen - len(sequence))
    return sequence

def create_dataloaders():
    vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_pre_split(training_data_path=TRAINING_DATA_PATH, validation_data_path=VALIDATION_DATA_PATH, testing_data_path=TESTING_DATA_PATH, name_scope=DATASET + "-HAN")
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(WORD2VEC_MODEL_PATH,name_scope=DATASET + "-HAN")
    vocab_size = len(vocabulary_word2index)

    X_train, Y_train = load_data_multilabel_pre_split(vocabulary_word2index, vocabulary_word2index_label, data_path=TRAINING_DATA_PATH, keep_label_percent=1.0)
    X_valid, Y_valid = load_data_multilabel_pre_split(vocabulary_word2index, vocabulary_word2index_label, data_path=VALIDATION_DATA_PATH, keep_label_percent=1.0)

    # pad and truncate each discharge notes to exactly 2500 tokens
    X_train_padded = [pad_or_truncate_sequence(x, SEQUENCE_LENGTH) for x in X_train]
    X_valid_padded = [pad_or_truncate_sequence(x, SEQUENCE_LENGTH) for x in X_valid]

    # Convert your data to PyTorch tensors
    # split X into sentences
    X_train_tensor = torch.tensor(X_train_padded)
    Y_train_tensor = torch.tensor(Y_train).float()

    X_valid_tensor = torch.tensor(X_valid_padded)
    Y_valid_tensor = torch.tensor(Y_valid).float()

    # Create a TensorDataset from your tensors
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)

    # Create a DataLoader from your dataset
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('shuffled training data')
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader, vocab_size

def initialization_using_word2vec(model):
    vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_pre_split(training_data_path=TRAINING_DATA_PATH, validation_data_path=VALIDATION_DATA_PATH, testing_data_path=TESTING_DATA_PATH, name_scope=DATASET + "-HAN")
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(WORD2VEC_MODEL_PATH,name_scope=DATASET + "-HAN")
    
    vocab_size = len(vocabulary_word2index)

    # initialize embeddings with pretrained word2vec embeddings
    emb_bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    embeddings = torch.rand((vocab_size, EMBED_SIZE)) * 2 * emb_bound - emb_bound
    
    word_w2v = Word2Vec.load(WORD2VEC_MODEL_PATH)
    for i in range(vocab_size):
        word = vocabulary_index2word[i]
        # word might not exists in the word2vec model!
        if word in word_w2v.wv.key_to_index:
            embeddings[i] = torch.tensor(word_w2v.wv[word])


    # initialize linear projection on layer
    label_w2v_400 = Word2Vec.load(WORD2VEC_LABEL_PATH_400)
    bound = np.sqrt(6.0) / np.sqrt(NUM_CLASSES + HIDDEN_SIZE * 4)
    W_linear = torch.rand((NUM_CLASSES, HIDDEN_SIZE*4)) * 2 * bound - bound

    for i in range(NUM_CLASSES):
        label = vocabulary_index2word_label[i]
        if label in label_w2v_400.wv.key_to_index:
            W_linear[i, :] = torch.tensor(label_w2v_400.wv[label])
        else:
            print(i, label, 'not in vocab')

    # initialize context vector
    label_w2v_200 = Word2Vec.load(WORD2VEC_LABEL_PATH_200)
    word_context_vector = torch.rand((NUM_CLASSES, HIDDEN_SIZE * 2)) * 2* bound - bound
    sentence_context_vector = torch.rand((NUM_CLASSES, HIDDEN_SIZE * 2)) * 2* bound - bound

    for i in range(NUM_CLASSES):
        label = vocabulary_index2word_label[i]
        if label in label_w2v_200.wv.key_to_index:
            word_context_vector[i, :] = torch.tensor(label_w2v_200.wv[label])
            sentence_context_vector[i, :] = torch.tensor(label_w2v_200.wv[label])
        else:
            print(i, label, 'not in vocab')

    with torch.no_grad():
        model.embeddings.weight.data.copy_(embeddings)
        model.word_attention.context_vector.data.copy_(word_context_vector)
        model.sentence_attention.context_vector.data.copy_(sentence_context_vector)
        model.W.data.copy_(W_linear)

    return model



def create_subset_dataloader(original_loader, num_samples=10):
    """
    Create a new DataLoader with a shuffled subset of the original data.

    Parameters:
    - original_loader (DataLoader): The original DataLoader.
    - num_samples (int): Number of samples to include in the subset.

    Returns:
    - DataLoader: A new DataLoader with the shuffled subset of data.
    """
    # Get the original dataset from the DataLoader
    original_dataset = original_loader.dataset

    # Check if the desired number of samples is not more than the dataset size
    num_samples = min(num_samples, len(original_dataset))

    # Generate a random permutation of indices and select the first num_samples indices
    shuffled_indices = torch.randperm(len(original_dataset))[:num_samples]
    subset = Subset(original_dataset, shuffled_indices)

    # Create a new DataLoader with the subset
    subset_loader = DataLoader(subset, batch_size=original_loader.batch_size, shuffle=False)

    return subset_loader

def get_micro_metrics_all_thresholds(all_logits, all_labels_actuals):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_probs = [F.sigmoid(logits) for logits in all_logits]
    ret = {}
    for threshold in thresholds:
        preds = np.concatenate(all_probs)
        actuals = np.concatenate(all_labels_actuals)
        preds = np.where(preds > threshold, 1, 0)
        acc, prec, rec, f1 = all_micro(preds.ravel(), actuals.ravel())
        ret[threshold] = (acc, prec, rec, f1)
    return ret