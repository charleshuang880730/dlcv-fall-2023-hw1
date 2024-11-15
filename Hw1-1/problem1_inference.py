# -*- coding: utf-8 -*-

# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader, Dataset
from problem1_model import Classifier

# This is for the progress bar.
from tqdm.auto import tqdm
import argparse

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
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
batch_size = 32
# The number of training epochs.
n_epochs = 70
# If no improvement in 'patience' epochs, early stop.
patience = 10
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

argparser = argparse.ArgumentParser()
argparser.add_argument("--input", type=str, help="Input file path.")
argparser.add_argument("--output", type=str, default="output.csv", help="Output file path.")
argparser.add_argument("--model", type=str, default="Part1B_resnet64x4_ver2_best.ckpt", help="Model file path.")
args = argparser.parse_args()

""" Dataloader for test """
# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = Dataset(f"{args.input}", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

""" Testing and generate prediction CSV """

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{args.model}"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

df = pd.DataFrame()
df["filename"] = [os.path.basename(file) for file in test_set.files]  # ensure only filenames, not full paths
df["label"] = prediction

df.to_csv(f"{args.output}", index=False, header=True)

