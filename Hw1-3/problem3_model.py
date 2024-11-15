import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

class Modified_VGG16_FCN32s(nn.Module):
    def __init__(self, n_class=7):
        super(Modified_VGG16_FCN32s, self).__init__()

        # Use pretrained VGG16 for feature extraction
        vgg = models.vgg16(weights= models.VGG16_Weights.DEFAULT)
        
        # Extract features from VGG16 (excluding the classifier part)
        self.backbone = vgg.features
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 2, 2, 0),  
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2, 0),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2, 0),    
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2, 0),    
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_class, 2, 2, 0),    
            nn.ReLU(),
        )
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x


class DeepLabV3_ResNet50(nn.Module):
    def __init__(self, n_class=7):
        super(DeepLabV3_ResNet50, self).__init__()
        
        # Use DeepLabv3 with ResNet50 backbone
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True, aux_loss=None)#, num_classes=n_class)
        self.model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.model(x)["out"]
        return x
