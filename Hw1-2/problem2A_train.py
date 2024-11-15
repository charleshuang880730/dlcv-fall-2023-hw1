# Import Packages"""

_exp_name = "Part2A_SSL_pretraining_"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from byol_pytorch import BYOL


# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, random_split
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models

# This is for the progress bar.
from tqdm.auto import tqdm
import random
# import wandb

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# wandb.init(project="DLCV_Hw1P2A", entity="charles0730")

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class Dataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(Dataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
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

# Part1: Pre-train ResNet50 backbone on Mini-ImageNet dataset via SSL
resnet = models.resnet50(pretrained=False)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
).to(device)

# Initialize a model, and put it on the device specified.
# model = learner.to(device)
# The number of batch size.
batch_size = 256
# The number of training epochs.
n_epochs = 1000
# If no improvement in 'patience' epochs, early stop.
patience = 30
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)

# wandb.watch(learner, log="all")

dataset = Dataset("../hw1_data/p2_data/mini/train", tfm=test_tfm)
train_set, valid_set = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# Pre-training
stale = 0
best_loss = np.inf

for epoch in range(n_epochs):

    train_loss = []

    for batch in tqdm(train_loader):

        images, labels = batch
        images = images.to(device)

        loss = learner(images)

        opt.zero_grad()

        loss.backward()

        opt.step()

        learner.update_moving_average() # update moving average of target encoder

        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    val_loss = []

    for batch in tqdm(valid_loader):

        # Forward & Backpropagation
            images, labels = batch
            images = images.to(device)

            with torch.no_grad():
                loss = learner(images)

            # Update
            loss = loss.item()
            val_loss.append(loss)

    valid_loss = sum(val_loss) / len(val_loss)

    # wandb.log({"Train Loss": train_loss})
    # wandb.log({"Validation Loss": valid_loss})

    # update logs
    if valid_loss < best_loss:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")

    # save models
    if valid_loss < best_loss:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(resnet.state_dict(), './resnet50_improved_ver2.pt') # only save best to prevent output memory exceed error
        best_loss = valid_loss
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
