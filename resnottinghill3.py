from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import scipy.io as scio
import numpy as np
import random
import torch
import os
from os import path
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import pdist
from munkres import Munkres
from scipy.spatial.distance import squareform
from sklearn.metrics.cluster import normalized_mutual_info_score


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def thrC(C,ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N,N))
		S = np.abs(np.sort(-np.abs(C),axis=0))
		Ind = np.argsort(-np.abs(C),axis=0)
		for i in range(N):
			cL1 = np.sum(S[:,i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while(stop == False):
				csum = csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
				t = t + 1
	else:
		Cp = C

	return Cp



def post_proC(C, K, d, alpha):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	C = 0.5*(C + C.T)
	r = d*K + 1
	U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
	U = U[:,::-1]
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis = 1)
	Z = U.dot(U.T)
	Z = Z * (Z>0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1
	return grp, L

def err_rate(gt_s, s):
	c_x = best_map(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate
data = scio.loadmat("/data/xyj/qyh/NottingHill_lite.mat")
data = data['trackStructSort']
imageCell = data['imageCell']
frame = data['frame']
label = data['personID']
imagecell = scio.loadmat("/data/xyj/qyh/NottingHill_lite.mat")['trackStructSort']['imageCell']
k = 0
f = np.zeros([np.size(data),np.size(data)])
for i in range(np.size(data)):
    for j in range(i+1,np.size(data)):
        if (frame[0][i][0][0]<=frame[0][j][0][0])and (frame[0][j][0][0]<=frame[0][i][0][np.size(imageCell[0][i])-1] )or( frame[0][i][0][0]>=frame[0][j][0][0])and (frame[0][i][0][0]<=frame[0][j][0][np.size(imageCell[0][j])-1]):
          f[i][j]=1
for i in range(np.size(data)):
    k = k+np.size(imageCell[0][i])
LABEL = np.zeros([k])
t = 0

w = 6
s = w
for i in range(np.size(data)):
    LABEL[t:t + np.size(imageCell[0][i])] = label[0][i]
    t = t + np.size(imageCell[0][i])
for i in range(np.size(data)):
    # a = np.arange(np.size(imageCell[0][i])).tolist()
    # flag = random.sample(a,10)
    # flag1 = np.zeros(np.size(imageCell[0][i]))
    # for k in range(5):
    #   # flag1[flag[k]] = 1;
    #   flag1[k] = 1
    imageCell[0][i] = imageCell[0][i][0][0:w]
    imageCell[0][i] = np.reshape(imageCell[0][i], [1,w])
image = imageCell[0][0]

for i in range(1, np.size(data)):
    image = np.concatenate((image, imageCell[0][i]), 1)


Img = np.zeros([image.shape[1], 60, 60, 3])
for i in range(image.shape[1]):
    Img[i, :, :, :] = image[0][i] / 255

Label = np.zeros([np.size(image)])
T = np.zeros([Img.shape[0]])
t = 0
for i in range(np.size(data)):
    Label[t:t + np.size(imageCell[0][i])] = label[0][i]
    T[t:t + np.size(imageCell[0][i])] = i
    t = t + np.size(imageCell[0][i])


F = np.zeros([Img.shape[0], Img.shape[0]])

for i in range(Img.shape[0]):
    for j in range(i + 1, np.size(data)):
        if(f[i][j]==1):
            F[i*s:i*s+s,j*s:j*s+s] = 1
            F[j*s:j*s+s,i*s:i*s+s] = 1
T1 = np.zeros([image.shape[1],image.shape[1]])
for i in range(image.shape[1]):
  for j in range(image.shape[1]):
    T1[i][j] = (1- T[i]==T[j])*1
# T1 = T1-np.diag(np.diag(T1))
# Track = squareform(T1)

F = torch.Tensor(F).to(device)
# G = F.mm(F.t()).to(device)
Track = torch.Tensor(T1).to(device)
# T1 = torch.Tensor(T1).cuda()

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
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128 ,64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.coef = nn.Parameter(torch.ones(batch_size, batch_size)*1e-4 , requires_grad=True)
        # self.coef1 = nn.Parameter(torch.ones(batch_size, batch_size)*1e-4 , requires_grad=True)
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

        return x

    def forward(self, x):
        z = self._forward_impl(x)
        b = z.size()
        z = z.view(self.batch_size, -1)
        z_ssc = (self.coef-torch.diag(torch.diag(self.coef))).mm(z)
        c = torch.reshape(z_ssc, b)
        x_r = self.convT(c)
        return x_r, z, z_ssc, self.coef


base_path = "/data/xyj/qyh/subspace_cluster"
train_time =3#2 #time1 1 7 0 0
def save_checkpoint(state, filename='checkpoint.pth.tar'):
  filefolder = "{}/model/parameter/res:{}".format(base_path, train_time)
  if not path.exists(filefolder):
    os.makedirs(filefolder)
  torch.save(state,path.join(filefolder,filename))





def train(model, Img, optimizer, reg1, reg2, reg3, reg4, T, F ,epoch):
    optimizer.zero_grad()
    x_r, z, z_ssc,coef = model(Img)
    recon = torch.sum(torch.pow(Img-x_r, 2))
    cost_ssc = torch.sum(torch.pow(z-z_ssc, 2))
    reg_ssc = torch.sum(torch.pow(coef, 2))
    # track_info = torch.sum(torch.nn.functional.pdist(coef,p=2)*T)
    track_info = torch.sum(T.mul(coef.mul(coef)))
    frame_info = torch.sum(F.mul(coef.mul(coef)))
    loss = recon + reg1*reg_ssc + reg2*cost_ssc +reg3*track_info + reg4*frame_info
    # loss = cost_ssc + reg1*recon
    loss.backward()
    # print(coef.grad)
    optimizer.step()
    print("epoch: %.1d" % epoch, "cost: %f" % (loss / float(batch_size)))
    # print(model.coef)
    return loss


Imgs = np.reshape(Img, (Img.shape[0], 3, 60, 60))
Imgs = torch.Tensor(Imgs)
resume = "/data/xyj/qyh/subspace_cluster/model/parameter/res_pretrain:6/checkpoint.pth.tar"
# resume = "/data/xyj/qyh/subspace_cluster/model/parameter/res_pretrain:3/checkpoint.pth.tar"
# resume = "/content/drive/subspace_clustering/model/parameter/res:16/checkpoint.pth.tar"
# resume = ""
resume_arg = True
epochs = 20000
start_epoch = 0
batch_size = np.size(image)
print(batch_size)
alpha = 0.34
learning_rate = 1e-3
num_class = 7

reg1 = 1#1 7 5 0
reg2 = 6
reg3 = 1
reg4 = 0.5
print("reg2:%.1d" % reg2)
print("reg3:%.1d" % reg3)
print("reg4:%.1d" % reg4)

model = ResNet(BasicBlock,[3,4,6],batch_size=batch_size)
# ignored_params = list(map(id, model.coef))
# base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
params_list = [{'params': model.convT.parameters(), 'lr': 1e-4},]
params_list.append({'params': model.coef, 'lr': 1e-3})
params_list.append({'params': model.layer1.parameters(), 'lr': 1e-4})
params_list.append({'params': model.layer2.parameters(), 'lr': 1e-4})
params_list.append({'params': model.layer3.parameters(), 'lr': 1e-4})
params_list.append({'params': model.conv1.parameters(), 'lr': 1e-4})
# if torch.cuda.device_count()>1:
#     print("Let's use",torch.cuda.device_count(),"GPUs!")
#     model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(params_list, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[40,80],gamma = 0.8)
Imgs = Imgs.to(device)
label_1 = np.zeros([np.size(LABEL)])
# coef = Variable(torch.ones([batch_size, batch_size]) * 1.0e-4, requires_grad=True)
# coef = coef.cuda().float()
a = np.zeros([100])
b = np.zeros([100])
c = np.zeros([100])
d = np.zeros([100])
k=0
print("=> loading checkpoint '{}'".format(resume))
checkpoint = torch.load(resume)
# model_dict = model.state_dict()
# state_dict = {o:v for o,v in checkpoint.items() if o in model_dict.keys()}
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))

print(model.conv1.parameters())
for epoch in range(1,epochs+1):
    loss = train(model, Imgs, optimizer, reg1, reg2, reg3, reg4, Track, F, epoch )
    # save_checkpoint({
    #       'epoch':epoch+1,
    #       'state_dict':model.state_dict(),

    #   })
    scheduler.step()
    if (epoch % 500 == 0) and (epoch >= 50):
        save_checkpoint({
            'epoch': epoch ,
            'state_dict': model.state_dict(),
            'coef': model.coef
        })
        print("epoch: %.1d" % epoch, "cost: %.8f" % (loss / float(batch_size)))
        coef = model.coef.detach().cpu().numpy()
        C = thrC(coef, alpha)
        acc = np.zeros([30])
        nmi = np.zeros([30])
        nmi_1 = np.zeros([30])
        for j in range(1,12):
          y_x, CKSym_x = post_proC(C, 7, j+1, 8)
          y_x = y_x.astype(np.int64)
          t = 0
          for i in range(np.size(imagecell)):
              label_1[t:t + np.size(imagecell[0][i])] = np.argmax(np.bincount(y_x[i * w:i * w + w]))
              t = t + np.size(imagecell[0][i])
          missrate_x = err_rate(LABEL, label_1)
          acc[j] = 1 - missrate_x
          nmi_1[j] = normalized_mutual_info_score(LABEL, label_1, average_method='arithmetic')
          label_2 = best_map(LABEL,label_1)
          nmi[j] = normalized_mutual_info_score(LABEL, label_2, average_method='arithmetic')
        a[k] = np.max(acc)
        b[k] = np.argmax(acc)+1
        c[k] = np.max(nmi)
        d[k] = np.max(nmi_1)
        k = k + 1
        print(a)
        print(b)
        print(c)
        print(d)
        print("reg2:%.1d" % reg2)
        print("reg3:%.1d" % reg3)
        print("reg4:%.1d" % reg4)
