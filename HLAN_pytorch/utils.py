import os
import pickle
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

from torch.utils.data import DataLoader, Subset

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


def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]

def create_vocabulary_label_pre_split(training_data_path,validation_data_path,testing_data_path,name_scope='',use_seq2seq=False,label_freq_th=0):
    '''
    create vocabulary from data split files - validation data path can be None or empty string if not exists.
    '''
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_label_vocabulary.pik"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        count=0
        vocabulary_word2index_label={}
        vocabulary_index2word_label={}
        vocabulary_label_count_dict={} #{label:count}
        
        # the list of data split files: not including validation data if it is set as None
        list_data_split_path = [training_data_path,validation_data_path,testing_data_path] if validation_data_path != None and validation_data_path != '' else [training_data_path,testing_data_path] 
        for data_path in list_data_split_path:
            print("create_vocabulary_label_sorted.started.data_path:",data_path)
            #zhihu_f_train = codecs.open(data_path, 'r', 'utf8')
            zhihu_f_train = codecs.open(data_path, 'r', 'latin-1')
            lines=zhihu_f_train.readlines()
            for i,line in enumerate(lines):
                if '__label__' in line:  #'__label__-2051131023989903826
                    label=line[line.index('__label__')+len('__label__'):].strip().replace("\n","")
                    # add multi-label processing
                    #print(label)
                    labels=label.split(" ")
                    for label in labels:
                        if label == '':
                            print('found empty label!') 
                            continue # this is a quick fix of the empty label problem, simply not recording it.
                        if vocabulary_label_count_dict.get(label,None) is not None:
                            vocabulary_label_count_dict[label]=vocabulary_label_count_dict[label]+1
                        else:
                            vocabulary_label_count_dict[label]=1
        list_label=sort_by_value(vocabulary_label_count_dict) # sort the labels by their frequency in the training dataset.

        print("length of list_label:",len(list_label));#print(";list_label:",list_label)
        countt=0

        ##########################################################################################
        if use_seq2seq:#if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list=[0,1,2];label_special_list=[_GO,_END,_PAD]
            for i,label in zip(i_list,label_special_list):
                vocabulary_word2index_label[label] = i
                vocabulary_index2word_label[i] = label
        #########################################################################################
        for i,label in enumerate(list_label):
            if i<10:
                count_value=vocabulary_label_count_dict[label]
                print("label:",label,"count_value:",count_value)
                countt=countt+count_value
            if vocabulary_label_count_dict[label]>=label_freq_th:
                indexx = i + 3 if use_seq2seq else i
                vocabulary_word2index_label[label]=indexx
                vocabulary_index2word_label[indexx]=label
        print("count top10:",countt)

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    print("create_vocabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label

def create_vocabulary(word2vec_model_path,name_scope=''):
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_word_vocabulary.pik"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        print("create vocabulary. word2vec_model_path:",word2vec_model_path)
        #model=word2vec.load(word2vec_model_path,kind='bin') # for danielfrg's word2vec models
        model = Word2Vec.load(word2vec_model_path) # for gensim word2vec models
        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'
        special_index=0
        if 'biLstmTextRelation' in name_scope:
            vocabulary_word2index['EOS']=1 # a special token for biLstTextRelation model. which is used between two sentences.
            vocabulary_index2word[1]='EOS'
            special_index=1
        # wv.vocab.keys() is no longer available in gensim 4
        # for i,vocab in enumerate(model.wv.vocab.keys()):
        for i,vocab in enumerate(model.wv.key_to_index.keys()):
            #if vocab == '':
            #    print(i,vocab)
            vocabulary_word2index[vocab]=i+1+special_index
            vocabulary_index2word[i+1+special_index]=vocab

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word), data_f)
    return vocabulary_word2index,vocabulary_index2word