import torch
import torch.nn as nn
import torchvision.models as models

class FCN16s(nn.Module):
    
    
    def __init__(self,num_classes = 1000):
        super(FCN16s,self).__init__()
        
        self.model = models.vgg16(pretrained=True)
        
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
                nn.Conv2d(512,4096,7,padding = 3),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(4096,4096,1,padding = 0),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(4096,num_classes,1,padding = 0))
        
        self.conv = nn.Conv2d(4096,num_classes,1)
        
        self.features_skip = nn.Sequential()
        
        self.features_rest = nn.Sequential()
        
        for i in range(24):
            self.features_skip.add_module('{}'.format(i),self.model.features[i])
        
        for i in range(24,31):
            self.features_rest.add_module('{}'.format(i),self.model.features[i])
        
        self.classifier[0].weight.data = (self.model.classifier[0].weight.data).view(4096,512,7,7)
        self.classifier[3].weight.data = (self.model.classifier[3].weight.data).view(4096,4096,1,1)
        self.classifier[0].bias.data = self.model.classifier[0].bias.data
        self.classifier[3].bias.data = self.model.classifier[3].bias.data
        self.conv.weight.data = torch.zeros(num_classes,512,1,1)
        self.conv.bias.data = torch.zeros(num_classes)
        
        if num_classes == 1000:

            self.classifier[6].weight.data = (self.model.classifier[6].weight.data).view(1000,4096,1,1)
            self.classifier[6].bias.data = self.model.classifier[6].bias.data
            
        else:
            
            self.classifier[6].weight.data = torch.randn(num_classes,4096,1,1)
            self.classifier[6].bias.data = torch.randn(num_classes)

        
    def forward(self,x):
        size = (x.shape[2], x.shape[3])
        x = self.features_skip(x)
        y = x
        y = self.conv(y)
        x = self.features_rest(x)
        x = self.classifier(x)
        x = nn.functional.upsample(x, size =(y.shape[2],y.shape[3]), mode='bilinear', align_corners=True) 
        return nn.functional.upsample((x+y), size=size, mode='bilinear', align_corners=True)