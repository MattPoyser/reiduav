

import torch
from torch import nn
import torchvision
from senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

            


#################################################

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=last_stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        
    def load_param(self, model_path):
        model_weight = torch.load(model_path)
        param_dict = model_weight['state_dict']
        new_state_dict = OrderedDict()
        
        for k, v in param_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
            
        for i in new_state_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(new_state_dict[i])
    
def resnet50_ibn_a(last_stride = 2, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(last_stride , Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50_ibn_b(last_stride,pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model









#####################################################

import torch
from torch import nn
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, stn_flag, model_name, pretrain_choice, backbone=None):
        super(Baseline, self).__init__()

        self.model_name = model_name
        print("model_name",model_name)
        self.isBackbone = False
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            resnet50 = torchvision.models.resnet50(pretrained=True)
            resnet50.layer4[0].conv2.stride = (1,1)
            resnet50.layer4[0].downsample[0].stride = (1,1)
            self.base = nn.Sequential(*list(resnet50.children())[:-2])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
                              
        elif model_name == 'mobilenetv3':
            self.in_planes = 960
            self.base = MobileNetV3_Large()

        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride = last_stride)
            print("in the right place")

        elif model_name == 'resnet50_ibn_a_old':
            self.base = resnet50_ibn_a_old(last_stride = last_stride)
            
        elif model_name == 'resnet50_ibn_b':
            self.base = resnet50_ibn_b(last_stride = last_stride)

        elif model_name == 'backbone':
            self.isBackbone = True
            if backbone is not None:
                self.base = backbone
        """
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        #self.fc_loc[2].weight.data.zero_()
        ##self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        """ 

        self.consensus = ConsensusModule("avg")
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        #self.stn_flag = stn_flag

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        
        
        
        
        
        
        #------------------for the new model----------------------
        self.middle_dim = 256 # middle layer dimension
        if self.isBackbone:
            self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1,1])
            self.backbone_type = str(type(backbone))
            if 'osnet' in self.backbone_type:
                self.additionallayer = nn.Conv2d(512, 2048, [1, 1])
            if 'hacnn' in self.backbone_type:
                self.additionallayerpre = nn.Conv2d(512, 1024, [1, 1])
                self.additionallayer = nn.Conv2d(1024, 2048, [1, 1])
            elif 'mlfn' in self.backbone_type:
                self.additionallayer = nn.Conv2d(1024, 2048, [1, 1])
            elif 'pcb' in self.backbone_type:
                self.additionallayer = nn.Conv2d(12288, 2048, [1, 1])
            elif 'mid' in self.backbone_type:
                self.additionallayer = nn.Conv2d(3072, 2048, [1, 1])
            else: # vit
                self.additionallayer = nn.Conv2d(768, 2048, [1, 1])
        else:
            self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [14,14])
            # self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [14,7]) #old, 224x112 images
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 
        #------------------for the new model----------------------
        if pretrain_choice == 'imagenet':
        
            print(model_path)
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        elif pretrain_choice == 'none':
            pass
        else:  #for our saved modle staaaaat with base.
            print(model_path)
            self.load_param2(model_path)
            print('Loading pretrained ImageNet model......')
        
   
    """    
    def forward(self, input):
        
        global_feat = self.gap(self.base(input.view((-1,3) + input.size()[-2:])))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        global_feat = self.bottleneck(global_feat)  # normalize for angular softmax
        cls_score = self.classifier(global_feat)
        
        y=global_feat.view((-1, 8) + global_feat.size()[1:])
        y=self.consensus(y)
        y=y.squeeze(1)
        cls_score=cls_score.view((-1, 8) + cls_score.size()[1:])
        cls_score=self.consensus(cls_score)
        #print("cls_score",cls_score.shape)
        cls_score=cls_score.squeeze(1)
        #print("cls_score after squeeze",cls_score.shape)      
            
        return cls_score,  y
    """
    
    def load_param(self, trained_path):
        print("in first pretran", trained_path)
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i not in self.state_dict() or 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
            
            
              
    def load_param2(self, model_path):
        model_weight = torch.load(model_path)
        param_dict = model_weight
        new_state_dict = OrderedDict()
        
        for k, v in param_dict.items():
            if(k[0:9]=="attention"):
                name = k
                #print("attention_conv")
            elif "bottleneck" in k:
                name = k
            else:   
                name = k[5:] # remove `module.`
            
            if(name[0:7]=="ization" or k[0:6]=="fc_loc"  or k[0:10]=="classifier"):
                continue
              
            else:    
                new_state_dict[name] = v
            
        for i in new_state_dict:
            """
            if "bottleneck" in i : 
                self.state_dict()[i].copy_(new_state_dict[i])
            """    
            if "attention_conv" in i:  
                 # print("i",i)
                 #self.attention_conv.state_dict().items().copy_(new_state_dict[i])
                 #state_dict["self." + "attention_conv"] = new_state_dict[i]  
                 self.state_dict()[i].copy_(new_state_dict[i])
                 
                 #print("attention_conv")   
            if "attention_tconv" in i:   
                 #self.attention_tconv.state_dict().items().copy_(new_state_dict[i])
                 self.state_dict()[i].copy_(new_state_dict[i])
                 
                 #state_dict["self." + "attention_tconv"] = new_state_dict[i] 
                 
                 #print("attention_tconv") 
              
            if i not in self.base.state_dict() or 'classifier' in i or  "attention_tconv" in i or "attention_conv" in i or "attention_tconv" in i or "bottleneck" in i:
                continue  
                
            self.base.state_dict()[i].copy_(new_state_dict[i])
            
            
            
            
    def forward (self, input, test=False):
        
        #if not self.training:input=input.view(input.size(0)//8,8,3,224, 112)
            
        b = input.size(0)
        t = input.size(1)
        global_feat = self.base(input.view((-1,3) + input.size()[-2:]))  # (b, 2048, 1, 1)
        # flatten to (bs, 2048)

        if self.isBackbone:
            if 'hacnn' in self.backbone_type and not test:
                global_feat = global_feat[1]
            if not test: # extract features and lose softmax results
                global_feat = global_feat[1]
            global_feat = global_feat.unsqueeze(2).unsqueeze(2)
            if 'hacnn' in self.backbone_type and not test:
                global_feat = self.additionallayerpre(global_feat) # todo fix do away with this
            global_feat = self.additionallayer(global_feat)
        a = F.relu(self.attention_conv(global_feat))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a_vals = a 
        
        a = F.softmax(a, dim=1)
        x=self.gap(global_feat)
        x =x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)

        f = att_x.view(b,self.in_planes)
        f = self.bottleneck(f) # normalize for angular softmax

        # raise AttributeError(f.shape, self.classifier)
        cls_score = self.classifier(f)
       
        
        if not self.training:
            return f
        
        return cls_score, f , a_vals




