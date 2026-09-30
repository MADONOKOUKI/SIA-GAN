import argparse
import os
import numpy as np
import math
import tensorboardX as tbx
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
from learnable_encryption import BlockScramble
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision    
from torch.nn.init import xavier_uniform_
import random
from Blockwise_scramble_LE import *#blockwise_scramble
from util_norm import total_variation_norm
parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")

parser.add_argument("--blockwise_scramble", type=str, default=False, help="interval betwen image samples")
parser.add_argument("--block_location_shuffle", type=str, default=True, help="interval betwen image samples")

parser.add_argument("--adaptation_network", type=str, default="None", help="interval betwen image samples")
parser.add_argument("--generator", type=str, default="two_convs", help="interval betwen image samples")
parser.add_argument("--discriminator", type=str, default="dcgan", help="interval betwen image samples")

parser.add_argument("--tensorboard_name", type=str, default="dcgan", help="interval betwen image samples")
parser.add_argument("--directory_name", type=str, default="dcgan", help="interval betwen image samples")

parser.add_argument("--adversarial_loss", type=str, default="original", help="interval betwen image samples")

parser.add_argument("--num_of_keys", type=int, default=1)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--mixup", type=bool, default=1)
parser.add_argument("--noise", type=bool, default=1)

opt = parser.parse_args()
args = parser.parse_args()

if not os.path.exists(opt.directory_name):
    os.mkdir(opt.directory_name)

perm = [i for i in range(args.num_of_keys)]
perm_pair = [(i,j) for i in range(args.num_of_keys) for j in range(args.num_of_keys)]

def mixup_data(x1, x2, y, alpha=args.alpha, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

 #   lam = 0.5

    mixed_x = lam * x1 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]

   # mixed_x = lam * x1 + (1 - lam) * x2
   # y_a, y_b = y, y

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def random_mask(x):
    mask = torch.zeros(32*32*3)
    mask[:32*16*3] = 1
    mask = mask[torch.randperm(32*32*3)]
    for i in range(x.size()[0]):
        x[i,:,:,:] = x[i,:,:,:] * mask.view(3,32,32) + (1-x[i,:,:,:]) * (1-mask.view(3,32,32))
        mask = mask[torch.randperm(32*32*3)]
    return x

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size*2, shuffle=True, num_workers=opt.n_cpu)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

adversarial_loss = torch.nn.BCELoss()
classififation_loss = torch.nn.CrossEntropyLoss()

#from mixup import *
import numpy
seed = 1234
#random.seed(seed)  
#numpy.random.seed(seed)  
#torch.manual_seed(seed)  
# cuda でのRNGを初期化  
#torch.cuda.manual_seed(seed) 

#from new_adaptation_nonshare_cnn_stgumbel import Generator
#from jigsaw_loop_4x4_size_mse_solve import Generator
#from jigsaw_loop_4x4_size_mse_solve_shuffle_tmp_ss import Generator
from generator import Generator
from dcgan_spec_acgan import Discriminator
from  no_adaptation_network import ShakePyramidNet

def gaussian(ins, is_training, stddev=0.1):
    if is_training:
        return ins + Variable(torch.randn(ins.size()) * stddev)
    return ins

generator = Generator().cuda()
discriminator = Discriminator().cuda()
net = ShakePyramidNet(depth=110, alpha=270, label=10).cuda()#net = net.to(device)

if cuda:
    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

_shf = [i for i in range(16)]
random.shuffle(_shf)
optimizer_G = torch.optim.Adam(list(generator.parameters())+list(net.parameters()),lr=1e-4)#, betas=(opt.b1, opt.b2))#eps=1e-8)#, lr = 1e-6)#, lr=1e-4, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(opt.b1, opt.b2))

writer = tbx.SummaryWriter(opt.tensorboard_name)

temp = 1.0
iteration = 0
from copy import deepcopy
temp_min = 0.5
ANNEAL_RATE = 0.00003
from torchvision.models.vgg import vgg19

for epoch in range(opt.n_epochs):
    generator.train()
    discriminator.train()
    net.train()
    a,b,c = 0,1,1
    temp = np.maximum(temp * np.exp(-ANNEAL_RATE*1), temp_min)
    for ii, (imgs, labels) in enumerate(trainloader):
        imgs1 = imgs.numpy().copy()
        imgs2 = imgs.numpy().copy()
        
        idx = np.random.randint(0,args.num_of_keys)
        if idx == 0:
          x_stack = blockwise_scramble(imgs1[imgs1.shape[0]//2:, :, :, :],perm[0])
          imgs1[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))
        else:
          x_stack = blockwise_decramble(imgs1[imgs1.shape[0]//2:, :, :, :],perm[0])
          imgs1[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))

        imgs1 = torch.from_numpy(imgs1)#.long()
        if args.mixup == True:
          if idx == 0:
            x_stack = blockwise_decramble(imgs2[imgs2.shape[0]//2:, :, :, :],perm[0])
            imgs2[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))
          else:
            x_stack = blockwise_scramble(imgs2[imgs2.shape[0]//2:, :, :, :],perm[0])
            imgs2[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))

         # x_stack = blockwise_scramble(imgs2[imgs2.shape[0]//2:, :, :, :],perm[np.random.randint(0,args.num_of_keys)])
         # imgs2[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))

          imgs2 = torch.from_numpy(imgs2)#.long()
          imgs1, imgs2, targets = imgs1.cuda(), imgs2.cuda(), labels[imgs.size(0)//2:imgs.size(0)].cuda()
          imgs[imgs.shape[0]//2:,:,:,:], targets_a, targets_b, lam = mixup_data(imgs1[imgs.shape[0]//2:,:,:,:], imgs2[imgs.shape[0]//2:,:,:,:], labels[imgs.size(0)//2:imgs.size(0)], args.alpha, True)
          imgs, targets_a, targets_b = map(Variable, (imgs, targets_a, targets_b))#(inputs, targets_a, targets_b))
        else:
          imgs = imgs1
        imgs[imgs.shape[0]//2:, :, :, :] = random_mask(imgs[imgs.shape[0]//2:, :, :, :])
        imgs ,labels = imgs.to(device),labels.to(device)
        real_imgs = (Variable(imgs[:imgs.shape[0]//2, :, :, :].type(Tensor))-0.5)/0.5
        z = (Variable(imgs[imgs.shape[0]//2:, :, :, :].type(Tensor))-0.5)/0.5
#        z = gaussian(z,True)
        ys = torch.zeros(imgs.size(0)//2, 10).scatter_(1, labels[imgs.size(0)//2:imgs.size(0)].view(imgs.size(0)//2,1).long().cpu(), 1).to(device)
        gen_imgs = generator(z,ys)
        outputs = net(gen_imgs)

        ce = nn.CrossEntropyLoss()

        judge_real, _, output_real = discriminator(real_imgs)
        judge_fake, _, output_scramble = discriminator(gen_imgs.detach())

        d_loss = nn.ReLU()(1.0 + judge_fake ).mean() + nn.ReLU()(1.0 - judge_real).mean()  #+ ce(output_real, labels) + ce(output_scramble, labels)
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        judge_fake, _, _ = discriminator(gen_imgs) 
        if args.mixup == True:
          g_loss = - judge_fake.mean() + mixup_criterion(ce, outputs, targets_a.to(device), targets_b.to(device), lam)
          _, predicted = outputs.max(1)
          total = targets.size(0)
          correct = (lam * predicted.eq(targets_a.to(device).data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.to(device).data).cpu().sum().float()).item()
        else:
          g_loss = - judge_fake.mean() + ce(outputs, labels[imgs.shape[0]//2:])
          _, predicted = outputs.max(1)
          total = labels[imgs.shape[0]//2:].size(0)
          correct = predicted.eq(labels[imgs.shape[0]//2:]).sum().item()

        if iteration % 1 == 0:
          optimizer_G.zero_grad()
          g_loss.backward()
          optimizer_G.step()

        batches_done = epoch * len(trainloader) + ii
        iteration = iteration + 1

        writer.add_scalars('data/train_loss',
        {
            'generator_loss': g_loss.item(),
            'discriminator_loss': d_loss.item(),
            'classification_loss': 100.*correct/total,
        }, (iteration + 1))

    discriminator.eval()
    generator.eval()
    net.eval()
    total_loss = 0
    mse = torch.nn.MSELoss()
    for i, (imgs, labels) in enumerate(testloader):
        save_image(imgs[0:16].data, opt.directory_name +"/answer%d.png" % epoch, nrow=5, normalize=True)

       # imgs1 = imgs.numpy()
        imgs = imgs.numpy().copy()

        x_stack = blockwise_scramble(imgs,perm[0])
        imgs  = np.transpose(x_stack,(0,3,1,2))

        imgs = torch.from_numpy(imgs)#.long()
        imgs = random_mask(imgs)
        imgs = (imgs - 0.5 ) / 0.5
        #imgs1 = torch.from_numpy(imgs1)
        #imgs1 = imgs1.to(device)
        imgs ,labels = imgs.to(device),labels.to(device)
        ys = torch.zeros(imgs.size(0), 10).scatter_(1, labels.view(imgs.size(0),1).long().cpu(), 1).to(device)
        gen_imgs  = generator(imgs,ys)#,1,temp)
        gen_imgs  = (gen_imgs + 1)/2.0
       # total_loss +=  mse(gen_imgs, imgs1)
        save_image(gen_imgs[0:16].data, opt.directory_name +"/%d.png" % epoch, nrow=5, normalize=True)
 #   print(total_loss.item())
writer.close()
  

