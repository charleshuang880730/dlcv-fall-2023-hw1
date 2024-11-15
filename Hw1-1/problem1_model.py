# Import necessary packages.
import torch
import torch.nn as nn

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        """# vggnet
        self.vgg = models.vgg19_bn(weights=True)
        self.vgg.classifier[6] = nn.Linear(in_features=4096, out_features=50)

        # efficientnet
        self.effnet = models.efficientnet_v2_l(weights=True)
        self.effnet.classifier = nn.Linear(1280, 50)"""

        # resnet 
        # self.resnet = models.resnet101(pretrained=True)
        self.resnet = models.resnext101_64x4d()
        in_features = self.resnet.fc.in_features
        num_classes = 50
        new_fc_layer = torch.nn.Linear(in_features, num_classes)
        self.resnet.fc = new_fc_layer


        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        """self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),     

            nn.Conv2d(256, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 1024, 3, 1, 1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 64, 3, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),       
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 50)
        )

        self.feature_extractor = nn.Sequential(*list(self.fc.children())[:-1])  # all layers except the last one
        self.classifier = list(self.fc.children())[-1]  # only the last layer"""

    """def forward(self, x, return_features=False):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        features = self.feature_extractor(out)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits"""

    def forward(self, x):
        
        """out = self.cnn(x)
        return self.fc(out)"""
    
        out = self.resnet(x)
        return out
    