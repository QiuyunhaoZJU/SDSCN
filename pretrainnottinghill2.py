import scipy.io as scio
import os
from os import path
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import pdist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path = "/data/xyj/qyh/subspace_cluster"
train_time = 2
def save_checkpoint(state, filename='checkpoint.pth.tar'):
  filefolder = "{}/model/parameter/res_pretrain:{}".format(base_path, train_time)
  if not path.exists(filefolder):
    os.makedirs(filefolder)
  torch.save(state,path.join(filefolder,filename))


data = scio.loadmat("/data/xyj/qyh/NottingHill_lite.mat")
data = data['trackStructSort']
imageCell = data['imageCell']
frame = data['frame']
for i in range(np.size(data)):
  # a = np.arange(np.size(imageCell[0][i])).tolist()
  # flag = random.sample(a,10)
  flag1 = np.zeros(np.size(imageCell[0][i]))
  for k in range(8):
    # flag1[flag[k]] = 1;
    flag1[k] = 1
  imageCell[0][i] = imageCell[0][i][0][flag1>0]
  imageCell[0][i] = np.reshape(imageCell[0][i],[1,8])



image = imageCell[0][0]
frames = frame[0][0]

for i in range(1, np.size(data)):
    image = np.concatenate((image, imageCell[0][i]), 1)


Img = np.zeros([image.shape[1],60,60,3])
for i in range(image.shape[1]):
  Img[i,:,:,:] = image[0][i]

Img = Img/255

print(np.shape(Img))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, batch_size, zero_init_residual=False,
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None):
            super(ResNet, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer
            self.batch_size = batch_size
            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.convT = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(inplace=True)
            )
            self.coef = nn.Parameter(torch.ones(batch_size, batch_size) * 1e-4, requires_grad=True)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return nn.Sequential(*layers)

    def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x

    def forward(self, x):
        z = self._forward_impl(x)
        x_r = self.convT(z)
        return x_r




def train(model, Img, optimizer, epoch):
    optimizer.zero_grad()
    x_r = model(Img)
    recon = 0.5*torch.sum(torch.pow(Img - x_r, 2))
    loss = recon
    loss.backward()
    optimizer.step()
    print("epoch: %.1d" % epoch, "cost: %f" % (loss / float(Img.shape[0])))
    return loss


Imgs = np.reshape(Img, (Img.shape[0], 3, 60, 60))
Imgs = torch.Tensor(Imgs)

# resume = "/data/xyj/qyh/subspace_cluster/model/parameter/res_pretrain:2/checkpoint.pth.tar"
resume = ""
resume_arg = True
epochs = 10000

batch_size = 2216
alpha = 0.04
learning_rate = 1e-3

start_epoch = 1
model = ResNet(BasicBlock,[3,4,6,3],batch_size=batch_size)
# if torch.cuda.device_count()>1:
#     print("Let's use",torch.cuda.device_count(),"GPUs!")
#     model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,5],gamma = 0.5)

Imgs = Imgs.to(device)

if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        if resume_arg:
            start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(resume))

for epoch in range(start_epoch, epochs+1):
    loss = train(model, Imgs, optimizer, epoch + 1)
    # save_checkpoint({
    #       'epoch':epoch+1,
    #       'state_dict':model.state_dict()
    #   })
    scheduler.step()
    if (epoch % 300 == 0):
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict()
        })



