import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from tools.ai.torch_utils import *

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model


#######################################################################

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, segmentation=False):
        super().__init__()

        self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class Classifier(Backbone):
    def __init__(self, model_name, num_classes=20):
        super().__init__(model_name, num_classes)
        
        self.BN = nn.ModuleList([nn.BatchNorm2d(2048) for _ in range(20)])
        self.stage6 = nn.ModuleList([nn.Conv2d(2048, 1, 1, bias=False) for _ in range(20)])
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes
        self.initialize([self.classifier, self.stage6, self.BN])
    
    def forward(self, x, with_cam=False):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)                  # x:[16, 2048, 32, 32]
        
        if with_cam:
            features = self.classifier(x)   # features:[16, 20, 32, 32]
            
            cams = make_cam(features)  # cams: [16, 20, 32, 32]

            img_feature_map = []
            
            for v in range(20):

                cam_class = cams[:, v, :, :]                               # cam_class: [16, 32, 32]
                mask_class = cam_class.unsqueeze(1).repeat(1, 2048, 1, 1)  # mask_class: [16, 2048, 32, 32]
                x_mask = x * mask_class                                    # x_mask: [16, 2048, 32, 32]
                x_mask = self.BN[v](x_mask)
                x_mask = self.stage6[v](x_mask)                     # x_mask:[16, 1, 32, 32]
                img_feature_map.append(x_mask)         

            img_feature_map = torch.cat(img_feature_map, dim=1)     # img_feature_map: [16, 20, 32, 32]
            logits = self.global_average_pooling_2d(img_feature_map)

            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True)
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits


