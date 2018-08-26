import torch
import torch.nn as nn
import torchvision.models as models
#kek
class FCN32s(nn.Module):
    
    
    def __init__(self,num_classes = 1000):
        super(FCN32s,self).__init__()
        
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
        
        self.classifier[0].weight.data = (self.model.classifier[0].weight.data).view(4096,512,7,7)
        self.classifier[3].weight.data = (self.model.classifier[3].weight.data).view(4096,4096,1,1)
        self.classifier[0].bias.data = self.model.classifier[0].bias.data
        self.classifier[3].bias.data = self.model.classifier[3].bias.data
        
        if num_classes == 1000:

            self.classifier[6].weight.data = (self.model.classifier[6].weight.data).view(1000,4096,1,1)
            self.classifier[6].bias.data = self.model.classifier[6].bias.data
            
        else:
            
            self.classifier[6].weight.data = torch.randn(num_classes,4096,1,1)
            self.classifier[6].bias.data = torch.randn(num_classes)
            
            
    def forward(self,x):
        size = (x.shape[2],x.shape[3])
        x = self.model.features(x)
        x = self.classifier(x)
        return nn.functional.upsample(x, size=size,mode='bilinear', align_corners=True)
    
