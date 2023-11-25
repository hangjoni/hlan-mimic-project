import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import string
import random

# import constants
from constants import *

# Add the parent directory to the system path
sys.path.append("..")

from HLAN.data_util_gensim import create_vocabulary_label_pre_split, create_vocabulary
from utils import load_data_multilabel_pre_split , create_dataloaders, initialization_using_word2vec
from HAN_model import HierarchicalAttentionNetwork

from metrics import *
from utils import get_micro_metrics_all_thresholds


def train(train_dataloader, valid_dataloader, vocab_size, epochs=1, lr=0.0005):
    model = HierarchicalAttentionNetwork(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, num_sentences=NUM_SENTENCES, sentence_length=SENTENCE_LENGTH, num_classes=NUM_CLASSES)
    model = initialization_using_word2vec(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    wandb.init(
        # set the wandb project where this run will be logged
        project="bd4h-project-v1",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "HLAN+LE",
        "dataset": "mimic-50",
        "loss": "BCEWithLogitsLoss"
        }
    )
    wandb.watch(model, criterion, log='all', log_freq=1)

    best_valid_loss = 100000 # arbitrarily high default
    
    dir_exist = True
    while dir_exist:
        random_string = ''.join(random.choice(string.ascii_letters) for i in range(5))
        checkpoint_dir = os.path.join("..", "checkpoints", random_string)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Created directory to save model checkpoint at {checkpoint_dir}")
            dir_exist = False

    for epoch in tqdm(range(epochs)):
        train_losses = []
        for i, (x, y) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            y_hat = model(x)
            train_loss = criterion(y_hat, y)
            train_loss.backward()
            train_losses.append(train_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            wandb.log({'step train loss': train_loss.item()})
        wandb.log({'epoch train loss': np.array(train_losses).mean()})

        with torch.no_grad():
            model.eval()
            valid_losses = []
            all_logits = []
            all_labels_actuals = []
            for i, (x, y) in tqdm(enumerate(valid_dataloader)):
                y_hat = model(x)
                valid_loss = criterion(y_hat, y)
                valid_losses.append(valid_loss.item())
                all_logits.append(y_hat)
                all_labels_actuals.append(y)
                wandb.log({'step valid loss': valid_loss.item()})
            
            ret = get_micro_metrics_all_thresholds(all_logits, all_labels_actuals)
            for threshold, (accuracy, precision, recall, f1) in ret.items():
                wandb.log({f'{threshold} accuracy': accuracy, f'{threshold} precision': precision, f'{threshold} recall': recall, f'{threshold} f1': f1})

            wandb.log({'epoch valid loss': np.array(valid_losses).mean()})
	    # Save checkpoint if valid loss improves
            if np.array(valid_losses).mean() < best_valid_loss:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_valid_loss.pt"))
                best_valid_loss = np.array(valid_losses.mean())
	
        model.train()
        scheduler.step(np.array(train_losses).mean())

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last.pt")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script, expect epochs and learning rate arguments, otherwise will train for 1 epoch with 0.0005 lr")
    parser.add_argument("--epochs", type="int", help = "Number of epochs to train for", default = 1)
    parser.add_argument("--lr", type="float", help = "Learning rate", default = 0.0005)
    args = parser.parse_args()
    train_dataloader, valid_dataloader, vocab_size = create_dataloaders()
    print(f"Training model with {args.epochs} epochs and {args.lr} learning rate")
    model = train(train_dataloader, valid_dataloader, vocab_size, epochs=args.epochs, lr=args.lr)
    print("Training done! Check Weights and Biases for plots. Check ../checkpoints folder for checkpoints.")
