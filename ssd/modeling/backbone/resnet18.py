import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ssd.modeling import registry

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if(downsample == True):
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias),
                                            nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if(self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
        
class ResNet18_SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64, 3, stride=1, padding=1), 
                                    BasicBlock(64, 64, 3, stride=1, padding=1))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 3, stride=2, padding=1, downsample=True), 
                                    BasicBlock(128, 128, 3, stride=1, padding=1))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 3, stride=2, padding=1, downsample=True), 
                                    BasicBlock(256, 256, 3, stride=1, padding=1))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 3, stride=2, padding=1, downsample=True), 
                                    BasicBlock(512, 512, 3, stride=1, padding=1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv5 = nn.Conv2d(512, 1024, 1)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)

    def forward(self, x):
        features = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        features.append(out)
        out = self.layer2(out)
        features.append(out)
        out = self.layer3(out)
        features.append(out)
        out = self.layer4(out)
        features.append(out)
        out = self.avgpool(out)
        out = self.conv5(out)
        features.append(out)
        return tuple(features)

@registry.BACKBONES.register('resnet18')
def vgg(cfg, pretrained=True):
    model = ResNet18_SSD(cfg)
    if pretrained:
        m = models.resnet18(pretrained=True)
        pretrained_dict = m.state_dict() 
        model_dict = model.state_dict() 
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model
