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
from Blockwise_scramble_LE import blockwise_scramble
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
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument("--mixup", type=float, default=0)
parser.add_argument("--random_pixel_inversion", type=float, default=0)
parser.add_argument("--block_scramble", type=float, default=0)
parser.add_argument("--random_pixel_inversion_test", type=float, default=0)

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

    mixed_x = lam * x1 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

torch.manual_seed(0)


transform_train = transforms.Compose([
 #   transforms.RandomCrop(32, padding=4),
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

from generator import Generator
from dcgan_spec_acgan import Discriminator
from  no_adaptation_network import ShakePyramidNet
import lpips

def random_mask(x):
    mask = torch.zeros(32*32*3)
    mask[:32*16*3] = 1
    mask = mask[torch.randperm(32*32*3)]
    for i in range(x.size()[0]):
        x[i,:,:,:] = x[i,:,:,:] * mask.view(3,32,32) + (1-x[i,:,:,:]) * (1-mask.view(3,32,32))
        mask = mask[torch.randperm(32*32*3)]
    return x

def gaussian(ins, is_training, stddev=0.1):
    if is_training:
        return ins + Variable(torch.randn(ins.size()) * stddev)
    return ins

generator = Generator().cuda()
discriminator = Discriminator().cuda()
net = ShakePyramidNet(depth=110, alpha=270, label=10).cuda()#net = net.to(device)
eval_lpips = lpips.LPIPS(net='alex')

if cuda:
    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()
    net = torch.nn.DataParallel(net).cuda()
    eval_lpips = torch.nn.DataParallel(eval_lpips).cuda()
    cudnn.benchmark = True

_shf = [i for i in range(16)]
random.shuffle(_shf)
optimizer_G = torch.optim.Adam(list(generator.parameters())+list(net.parameters()),lr=1e-4, betas=(0.5, 0.9))#, betas=(opt.b1, opt.b2))#eps=1e-8)#, lr = 1e-6)#, lr=1e-4, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.0, 0.9)) #(opt.b1, opt.b2))

writer = tbx.SummaryWriter(opt.tensorboard_name)

def total_variation_norm(input_matrix, beta= 2 ):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        B, H, W, C = input_matrix.size()
        to_check = input_matrix[:,0, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:,0, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:,0, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        to_check = input_matrix[:,1, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:,1, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:,1, :-1, 1:]  # Trimmed: top - right
        total_variation += (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        to_check = input_matrix[:,2, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:,2, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:,2, :-1, 1:]  # Trimmed: top - right
        total_variation += (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return  total_variation / (H*W*C)

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
        if args.block_scramble == 1: 
          x_stack = blockwise_scramble(imgs1[imgs1.shape[0]//2:, :, :, :],perm[np.random.randint(0,args.num_of_keys)])
          imgs1[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))

        imgs1 = torch.from_numpy(imgs1)
        if args.mixup == 1:
          if args.block_scramble == 1:
            x_stack = blockwise_scramble(imgs2[imgs2.shape[0]//2:, :, :, :],perm[np.random.randint(0,args.num_of_keys)])
            imgs2[imgs.shape[0]//2:,:,:,:] = np.transpose(x_stack,(0,3,1,2))

          imgs2 = torch.from_numpy(imgs2)
          imgs1, imgs2, targets = imgs1.cuda(), imgs2.cuda(), labels[imgs.size(0)//2:imgs.size(0)].cuda()
          imgs[imgs.shape[0]//2:,:,:,:], targets_a, targets_b, lam = mixup_data(imgs1[imgs.shape[0]//2:,:,:,:], imgs2[imgs.shape[0]//2:,:,:,:], labels[imgs.size(0)//2:imgs.size(0)], args.alpha, True)
          imgs, targets_a, targets_b = map(Variable, (imgs, targets_a, targets_b))
        else:
          imgs = imgs1
        if args.random_pixel_inversion == 1:
          imgs[imgs.shape[0]//2:, :, :, :] = random_mask(imgs[imgs.shape[0]//2:, :, :, :])

        imgs ,labels = imgs.to(device),labels.to(device)
        real_imgs = (Variable(imgs[:imgs.shape[0]//2, :, :, :].type(Tensor))-0.5)/0.5
        z = (Variable(imgs[imgs.shape[0]//2:, :, :, :].type(Tensor))-0.5)/0.5

        save_image((real_imgs[0:16].data + 1.0)/2.0, opt.directory_name +"/train_input%d.png" % epoch, nrow=5, normalize=True)
        save_image((z[0:16].data + 1.0)/2.0, opt.directory_name +"/train_realimg%d.png" % epoch, nrow=5, normalize=True)

        ys = torch.zeros(imgs.size(0)//2, 10).scatter_(1, labels[imgs.size(0)//2:imgs.size(0)].view(imgs.size(0)//2,1).long().cpu(), 1).to(device)
        gen_imgs, feature = generator(z,ys)
        outputs = net(gen_imgs)
        save_image((gen_imgs[0:16].data+ 1.0)/2.0, opt.directory_name +"/train_output%d.png" % epoch, nrow=5, normalize=True)

        ce = nn.CrossEntropyLoss()

        judge_real, _, output_real = discriminator(real_imgs)
        judge_fake, _, output_scramble = discriminator(gen_imgs.detach())

        d_loss = nn.ReLU()(1.0 + judge_fake ).mean() + nn.ReLU()(1.0 - judge_real).mean() 
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        judge_fake, _, _ = discriminator(gen_imgs) 
        if args.mixup == 1:
          g_loss = - judge_fake.mean() + mixup_criterion(ce, outputs, targets_a.to(device), targets_b.to(device), lam) +  1e-3 * total_variation_norm(feature)
          _, predicted = outputs.max(1)
          total = targets.size(0)
          correct = (lam * predicted.eq(targets_a.to(device).data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.to(device).data).cpu().sum().float()).item()
        else:
          g_loss = - judge_fake.mean() + ce(outputs, labels[imgs.shape[0]//2:]) +  1e-3 * total_variation_norm(feature)
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
    total_lpips_loss = 0
    for i, (imgs, labels) in enumerate(testloader):
        save_image(imgs[0:16].data, opt.directory_name +"/answer%d.png" % epoch, nrow=5, normalize=True)
        imgs_tmp = imgs.clone()

        imgs = imgs.numpy().copy()
        if args.block_scramble == 1:
          x_stack = blockwise_scramble(imgs,perm[0])
          imgs  = np.transpose(x_stack,(0,3,1,2))

        imgs = torch.from_numpy(imgs)
        if args.random_pixel_inversion_test == 1:
          imgs = random_mask(imgs)
        imgs = (imgs - 0.5) / 0.5
        save_image((imgs[0:16].data + 1.0)/2.0, opt.directory_name +"/test_input%d.png" % epoch, nrow=5, normalize=True)
 
        imgs ,labels = imgs.to(device),labels.to(device)
        ys = torch.zeros(imgs.size(0), 10).scatter_(1, labels.view(imgs.size(0),1).long().cpu(), 1).to(device)
        gen_imgs , feature = generator(imgs,ys)
        gen_imgs  = (gen_imgs + 1.0)/2.0

        save_image(gen_imgs[0:16].data, opt.directory_name +"/test_output%d.png" % epoch, nrow=5, normalize=True)
        for j in range(gen_imgs.size()[0]):
            total_lpips_loss = total_lpips_loss + eval_lpips.forward(imgs_tmp[j].view(1,3,32,32), gen_imgs[j].view(1,3,32,32)).sum().cpu().data
    print(total_lpips_loss / 10000 )

writer.close()
  

