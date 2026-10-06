import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ Bloc avec 'Saut de niveau' pour éviter la perte de gradient """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # <--- LE FAMEUX SAUT DE NIVEAU
        out = self.relu(out)
        return out

class DeepPneumoniaCNN(nn.Module):
    def __init__(self):
        super(DeepPneumoniaCNN, self).__init__()
        
        # 1. Préparation (224x224 -> 112x112)
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 2. Extracteur profond avec blocs résiduels (5 divisions spatiales au total)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResidualBlock(64), nn.MaxPool2d(2, 2) # -> 56x56
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResidualBlock(128), nn.MaxPool2d(2, 2) # -> 28x28
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResidualBlock(256), nn.MaxPool2d(2, 2) # -> 14x14
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            ResidualBlock(512), nn.MaxPool2d(2, 2) # -> 7x7
        )
        
        # 3. Classifieur robuste
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Effet Bagging
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
            # PAS DE SIGMOID ICI pour la stabilité mathématique de PyTorch
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

class UNet(nn.Module):
    # (Reste inchangé pour tes futurs tests)
    def __init__(self):
        super(UNet, self).__init__()
        self.dummy_layer = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.dummy_layer(x))