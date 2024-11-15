# -*- coding: utf-8 -*-
# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code
_exp_name = "Part1B_64x4_ver2_"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

# This is for the progress bar.
from tqdm.auto import tqdm
import random
# import wandb
from problem1_model import Classifier

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# wandb.init(project="DLCV_Hw1P1", entity="charles0730")

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((32, 32)),
    # You may add some transforms here.
    # transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(p=0.5),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class Dataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(Dataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im,label
        

"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
# The number of batch size.
batch_size = 128
# The number of training epochs.
n_epochs = 100
# If no improvement in 'patience' epochs, early stop.
patience = 100
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
# Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# wandb.watch(model, log="all")

"""# Dataloader"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = Dataset("../hw1_data/p1_data/train_50", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = Dataset("../hw1_data/p1_data/val_50", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

"""# Start Training"""

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    all_features = []  # 用於存儲所有的特徵
    all_labels = []  # 用於存儲所有的標籤

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            # logits = model(imgs.to(device))
            logits, features = model(imgs.to(device), return_features=True)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 在特定的epoch執行PCA和t-SNE視覺化
    if epoch in [0, n_epochs//2, n_epochs-1]:
        # For PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_features)

        plt.figure(figsize=(10, 5))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=all_labels, cmap='jet')
        plt.colorbar()
        plt.title(f'PCA Visualization - Epoch {epoch}')
        plt.savefig(f"PCA_Epoch_{epoch}.png")
        plt.show()

        # For t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(all_features)

        plt.figure(figsize=(10, 5))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_labels, cmap='jet')
        plt.colorbar()
        plt.title(f't-SNE Visualization - Epoch {epoch}')
        plt.savefig(f"tSNE_Epoch_{epoch}.png")
        plt.show()

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc})
    # wandb.log({"Validation Loss": valid_loss, "Validation Accuracy": valid_acc})

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

    scheduler.step()

# wandb.finish()
