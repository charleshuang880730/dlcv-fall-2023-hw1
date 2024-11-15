# -*- coding: utf-8 -*-

# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import imageio
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from problem3_model import Modified_VGG16_FCN32s, DeepLabV3_ResNet50
import viz_mask

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
import torchvision.transforms.functional as F
from torchvision.transforms.functional import hflip, vflip

# This is for the progress bar.
from tqdm.auto import tqdm
import random
import wandb


myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


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
patience = 15
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model_path, image_path):
    model = DeepLabV3_ResNet50().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    prediction = torch.argmax(output, dim=1).cpu().numpy()

    return prediction[0]

for model_path in ["model_0.pt", "model_9.pt", "model_19.pt"]:
    for image_path in ["../hw1_data/p3_data/validation/0013_sat.jpg", "../hw1_data/p3_data/validation/0062_sat.jpg", "../hw1_data/p3_data/validation/0104_sat.jpg"]:
        
        cmap = viz_mask.cls_color
        prediction = predict_image(model_path, image_path)
        
        seg_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls, color in viz_mask.cls_color.items():
            seg_img[prediction == cls] = color

        img = Image.open(image_path)  # Load the original image for visualization
        img = np.array(img)
        masks = viz_mask.read_masks(seg_img)

        cs = np.unique(masks)

        filename = os.path.basename(image_path)

        for c in cs:
            mask = np.zeros((img.shape[0], img.shape[1]))
            ind = np.where(masks == c)
            mask[ind[0], ind[1]] = 1
            img = viz_mask.viz_data(img, mask, color=cmap[c])
            imageio.imsave(f'output_{model_path}_{filename}.png', np.uint8(img))

        