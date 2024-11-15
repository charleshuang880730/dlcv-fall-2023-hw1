# -*- coding: utf-8 -*-

# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import imageio
import glob
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from problem3_model import DeepLabV3_ResNet50
import viz_mask

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torchvision.transforms.functional as F
from torchvision.transforms.functional import hflip, vflip

import random
import argparse
from tqdm.auto import tqdm


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
        # self.mask_files = sorted(glob.glob(os.path.join(path, "*_mask.png")))
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
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        if not isinstance(mask, torch.Tensor):
            mask = F.to_tensor(mask)

        normalize = transforms.Normalize(mean, std)
        img = normalize(img)

        return img, mask
    
    def validation_transform(self, img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Convert to Tensor
        img = F.to_tensor(img)
        # mask = F.to_tensor(mask)

        # Normalize (for ResNet)
        normalize = transforms.Normalize(mean, std)
        img = normalize(img)

        return img #, mask


    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # mask_path = self.mask_files[idx]

        img = Image.open(img_path)
        """mask = Image.open(mask_path)
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

        return img, mask"""

        img = self.validation_transform(img) # 傳入 None 作為 mask，因為我們不再需要它
        
        return img  # 返回圖像和空的 label，因為我們不再需要 mask


"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
model = DeepLabV3_ResNet50().to(device)
# The number of batch size.
batch_size = 5
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


argparser = argparse.ArgumentParser()
argparser.add_argument("--input", type=str, help="Input file path.")
argparser.add_argument("--output", type=str, default="output.csv", help="Output file path.")
argparser.add_argument("--model", type=str, default="Part3_DeepLabv3_best.ckpt", help="Model file path.")
args = argparser.parse_args()

"""# Dataloader"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
valid_set = Dataset(f"{args.input}", task="test")
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model_path = args.model

# Load the best model
model = DeepLabV3_ResNet50().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

global_idx = 0  

for images in tqdm(valid_loader):
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    for prediction in predictions:
        cmap = viz_mask.cls_color
        seg_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls, color in viz_mask.cls_color.items():
            seg_img[prediction == cls] = color

        filename = os.path.basename(valid_set.img_files[global_idx]).replace('_sat.jpg', '_mask.png') # 使用全局索引
        imageio.imsave(os.path.join(args.output, filename), seg_img)
        
        global_idx += 1  
        
"""for images, _ in tqdm(valid_loader):  # Note: We don't need labels here, so use _
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Process each image in the batch
    for idx, prediction in enumerate(predictions):
        cmap = viz_mask.cls_color
        seg_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls, color in viz_mask.cls_color.items():
            seg_img[prediction == cls] = color

        # Get the original image
        img = Image.open(valid_set.img_files[idx])
        img = np.array(img)
        masks = viz_mask.read_masks(seg_img)

        cs = np.unique(masks)

        filename = os.path.basename(valid_set.img_files[idx])
        for c in cs:
            mask = np.zeros((img.shape[0], img.shape[1]))
            ind = np.where(masks == c)
            mask[ind[0], ind[1]] = 1
            img = viz_mask.viz_data(img, mask, color=cmap[c])
            imageio.imsave(os.path.join(args.output, filename), np.uint8(img))"""        

