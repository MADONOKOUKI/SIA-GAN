import torch
import torch.nn as nn
from torch.autograd import Variable

class SpectralNorm(nn.Module):
    """Spectral normalization of weight with power iteration
    """
    def __init__(self, module, args, niter=1):
        super().__init__()
        self.module = module
        self.sn = args.sn
        self.niter = niter

        self.init_params(module)

    @staticmethod
    def init_params(module):
        """u, v, W_sn
        """
        w = module.weight
        height = w.size(0)
        width = w.view(w.size(0), -1).shape[-1]

        u = nn.Parameter(torch.randn(height, 1), requires_grad=False)
        v = nn.Parameter(torch.randn(1, width), requires_grad=False)
        module.register_buffer('u', u)
        module.register_buffer('v', v)

    @staticmethod
    def update_params(module, niter):
        u, v, w = module.u, module.v, module.weight
        height = w.size(0)

        for i in range(niter):  # Power iteration
            v = w.view(height, -1).t() @ u
            v /= (v.norm(p=2) + 1e-12)
            u = w.view(height, -1) @ v
            u /= (u.norm(p=2) + 1e-12)

        w.data /= (u.t() @ w.view(height, -1) @ v).data  # Spectral normalization

    def forward(self, x):
        if self.sn:
            self.update_params(self.module, self.niter)
        return self.module(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from spectral import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        class Args:
          sn = True
          ndf = 1024
          m_g = 4
        args = Args()
        self.args = args
        m_g = args.m_g
        ch = args.ndf

        self.layer1 = self.make_layer(3, ch//8)
        self.layer2 = self.make_layer(ch//8, ch//4)
        self.layer3 = self.make_layer(ch//4, ch//2)
        self.layer4 = SpectralNorm(nn.Conv2d(ch//2, ch, 3, 1, 1), self.args)
        self.linear = SpectralNorm(nn.Linear(ch*m_g*m_g, 1), self.args)
        self.real = SpectralNorm(nn.Linear(ch*m_g*m_g, 10), self.args)
        self.fake = SpectralNorm(nn.Linear(ch*m_g*m_g, 10), self.args)
        self.attn1 = Self_Attn(ch//2, 'relu')
        self.attn2 = Self_Attn(ch, 'relu')
        self.softmax  = nn.Softmax(dim=-1) #

    def make_layer(self, in_plane, out_plane):
        return nn.Sequential(
            SpectralNorm(
                nn.Conv2d(in_plane, out_plane, 3, 1, 1), self.args
            ),
            nn.LeakyReLU(0.1),
            SpectralNorm(
                nn.Conv2d(out_plane, out_plane, 4, 2, 1), self.args
            ),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out,p1 = self.attn1(out)
        out = self.layer4(out)
        out,p1 = self.attn2(out)
        out2 = out.view(out.size(0), -1)
        out1 = self.linear(out2)
        real = self.softmax(self.real(out2))
        fake = self.softmax(self.fake(out2))
        return out1.squeeze(), out2.squeeze(), fake.squeeze()

