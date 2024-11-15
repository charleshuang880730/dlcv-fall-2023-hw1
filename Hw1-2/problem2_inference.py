# Import Packages"""

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader, Dataset

# This is for the progress bar.
from tqdm.auto import tqdm
import argparse
from problem2B_model import Classifier

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)



# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

"""class Dataset(Dataset):

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

        return im,label"""

class Dataset(Dataset):
    def __init__(self, csv_file, img_dir, tfm=test_tfm):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = tfm

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = Image.open(img_name)
        image = self.transform(image)
        label = self.img_labels.iloc[idx, 2] if not pd.isna(self.img_labels.iloc[idx, 2]) else -1

        return image, label


"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
# The number of batch size.
batch_size = 32
# The number of training epochs.
n_epochs = 300
# If no improvement in 'patience' epochs, early stop.
patience = 10
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_csv", type=str, help="Input test.csv file path.")
argparser.add_argument("--input_img", type=str, help="Input image file path.")
argparser.add_argument("--output", type=str, default="output.csv", help="Output file path.")
argparser.add_argument("--model", type=str, default="Part2B_resnet50_office_MY_best.ckpt", help="Model file path.")
args = argparser.parse_args()

""" Dataloader for test """

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = Dataset(csv_file=f"{args.input_csv}", img_dir=f"{args.input_img}", tfm=test_tfm)
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

"""df = pd.DataFrame()
df["filename"] = [os.path.basename(file) for file in test_set.files]  # ensure only filenames, not full paths
df["label"] = prediction"""

df = pd.DataFrame({
    "id": test_set.img_labels["id"],
    "filename": test_set.img_labels["filename"],
    "label": prediction
})

df.to_csv(f"{args.output}", index=False, header=True)

""" Testing and generate prediction CSV """

"""model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{args.model}"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

# Create test csv
df = pd.read_csv(args.input_csv)
df["label"] = prediction
df.to_csv(args.output, index=False, header=True)"""