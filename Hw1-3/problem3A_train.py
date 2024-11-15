# -*- coding: utf-8 -*-

# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code
_exp_name = "Part3_VGG16FCN32s_"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from problem3_model import Modified_VGG16_FCN32s
import mean_iou_evaluate as m

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
import torchvision.transforms.functional as F
from torchvision.transforms.functional import hflip, vflip

# This is for the progress bar.
from tqdm.auto import tqdm
import random
# import wandb

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# wandb.init(project="DLCV_Hw1P3A", entity="charles0730")


"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""
# Reference : ChatGPT
class Dataset(Dataset):

    def __init__(self, path, task="test"):
        super(Dataset, self).__init__()

        self.img_files = sorted(glob.glob(os.path.join(path, "*_sat.jpg")))
        self.mask_files = sorted(glob.glob(os.path.join(path, "*_mask.png")))
        self.task = task


    def __len__(self):
        return len(self.img_files)
    
    def training_transform(self, img, mask):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Random Horizontal Flip
        if random.random() > 0.5:
            img, mask = hflip(img), hflip(mask)
        
        # Random Vertical Flip
        if random.random() > 0.5:
            img, mask = vflip(img), vflip(mask)

        # Convert to Tensor
        # img, mask = F.to_tensor(img), F.to_tensor(mask)
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        if not isinstance(mask, torch.Tensor):
            mask = F.to_tensor(mask)

        # Normalize (for ResNet)
        normalize = transforms.Normalize(mean, std)
        img = normalize(img)

        return img, mask
    
    def validation_transform(self, img, mask):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Convert to Tensor
        img, mask = F.to_tensor(img), F.to_tensor(mask)

        # Normalize (for ResNet)
        normalize = transforms.Normalize(mean, std)
        img = normalize(img)

        return img, mask


    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        raw_mask = mask.copy()

        mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
        mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
        mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
        mask[raw_mask == 2] = 3  # (Green: 010) Forest land
        mask[raw_mask == 1] = 4  # (Blue: 001) Water
        mask[raw_mask == 7] = 5  # (White: 111) Barren land
        mask[raw_mask == 0] = 6  # (Black: 000) Unknown
        mask = torch.tensor(mask)

        if self.task == "train":
            img, mask = self.training_transform(img, mask)
        else:
            img, mask = self.training_transform(img, mask)

        return img, mask


"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
model = Modified_VGG16_FCN32s().to(device)
# The number of batch size.
batch_size = 32
# The number of training epochs.
n_epochs = 100
# If no improvement in 'patience' epochs, early stop.
patience = 10
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# wandb.watch(model, log="all")

"""# Dataloader"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = Dataset("../hw1_data/p3_data/train", task="train")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = Dataset("../hw1_data/p3_data/validation", task="test")
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

"""# Start Training"""

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_loss = np.inf
best_miou = -1

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, mask = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, mask.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Record the loss and accuracy.
        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    all_preds = []
    all_gt = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, mask = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))


        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, mask.to(device))

        # Record the loss and accuracy.
        valid_loss.append(loss.item())

        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        y = mask.detach().cpu().numpy().astype(np.int64)
        all_preds.append(pred)
        all_gt.append(y)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    mIoU = m.mean_iou_score(np.concatenate(all_preds, axis=0), np.concatenate(all_gt, axis=0))

    # wandb.log({"Train Loss": train_loss})
    # wandb.log({"Validation Loss": valid_loss, "Validation mIoU": mIoU})

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, mIoU = {mIoU:.3f}")

    # update logs
    if mIoU < best_miou:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")


    # save models
    if mIoU < best_miou:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_miou = mIoU
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

    scheduler.step()

# wandb.finish()
