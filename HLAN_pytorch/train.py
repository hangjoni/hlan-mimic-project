import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import string
import random
import datetime

# import constants
from constants import *

# Add the parent directory to the system path
sys.path.append("..")

from utils import create_dataloaders, initialization_using_word2vec, create_subset_dataloader
from HAN_model import HierarchicalAttentionNetwork

from metrics import *


def train(train_dataloader, valid_dataloader, vocab_size, epochs=1, lr=0.0005, log=True, verbose=False, checkpoint_to_resume_from=None, device=torch.device("cpu")):
    model = HierarchicalAttentionNetwork(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, num_sentences=NUM_SENTENCES, sentence_length=SENTENCE_LENGTH, num_classes=NUM_CLASSES)
    if checkpoint_to_resume_from is None:
        model = initialization_using_word2vec(model)
    else:
        # load checkpoint. this is useful for resuming training
        model.load_state_dict(torch.load(checkpoint_to_resume_from))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    if log:
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
    date_time_string = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    while dir_exist:
        random_string = ''.join(random.choice(string.ascii_letters) for i in range(5))
        checkpoint_dir = os.path.join("..", "checkpoints", f"{date_time_string}_{epochs}epochs_{random_string}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Created directory to save model checkpoint at {checkpoint_dir}")
            dir_exist = False

    for epoch in tqdm(range(epochs)):
        train_losses = []
        valid_losses = []
        valid_logits = []
        valid_labels_actuals = []

        for i, (x, y) in tqdm(enumerate(train_dataloader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            train_loss = criterion(y_hat, y)
            train_loss.backward()
            train_losses.append(train_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step() 

        with torch.no_grad():
            model.eval()
            for i, (x, y) in tqdm(enumerate(valid_dataloader)):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                valid_loss = criterion(y_hat, y)
                valid_losses.append(valid_loss.item())
                # we use cpu to calculate metrics
                valid_logits.append(y_hat.to("cpu"))
                valid_labels_actuals.append(y.to("cpu"))

        acc, prec, rec, f1 = get_micro_metrics(valid_logits, valid_labels_actuals)
        if log:
            wandb.log({'epoch train loss': np.array(train_losses).mean()})
            wandb.log({'epoch valid loss': np.array(valid_losses).mean()})
            wandb.log({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

        if verbose:
            print(f"Epoch {epoch} train loss: {np.array(train_losses).mean()}, valid loss: {np.array(valid_losses).mean()}, f1: {f1}")

        # Save checkpoint if valid loss improves
        if np.array(valid_losses).mean() < best_valid_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_valid_loss.pt"))
            best_valid_loss = np.array(valid_losses).mean()

        model.train()
        scheduler.step(np.array(train_losses).mean())

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last.pt"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script, expect epochs and learning rate arguments, otherwise will train for 1 epoch with 0.0005 lr")
    parser.add_argument("--epochs", type=int, help = "Number of epochs to train for", default = 1)
    parser.add_argument("--lr", type=float, help = "Learning rate", default = 0.0005)
    parser.add_argument("--log", type=bool, help = "Whether to log to wandb", default = True)
    parser.add_argument("--verbose", type=bool, help = "Whether to print training progress", default = False)
    parser.add_argument("--checkpoint_to_resume_from", type=str, help = "Path to checkpoint to resume training from", default = None)
    parser.add_argument("--device", type=str, help = "Device to train on", default = "cpu")
    parser.add_argument("--mini", type=bool, help = "Whether to use mini dataset. Useful for testing", default = False)
    
    args = parser.parse_args()
    train_dataloader, valid_dataloader, vocab_size = create_dataloaders()
    if args.mini:
        train_dataloader = create_subset_dataloader(train_dataloader, 10)
        valid_dataloader = create_subset_dataloader(valid_dataloader, 10)
    if args.checkpoint_to_resume_from is not None:
        print(f"Resuming training from checkpoint {args.checkpoint_to_resume_from}")
    print(f"Training model with {args.epochs} epochs and {args.lr} learning rate")
    model = train(train_dataloader, valid_dataloader, vocab_size, epochs=args.epochs, lr=args.lr, log=args.log, verbose=args.verbose, checkpoint_to_resume_from=args.checkpoint_to_resume_from, device=torch.device(args.device))
    print("Training done! Check Weights and Biases for plots. Check ../checkpoints folder for checkpoints.")
