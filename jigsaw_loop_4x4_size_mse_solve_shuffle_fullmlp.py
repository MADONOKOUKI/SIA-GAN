from torch import nn
import torch
from linear import OriginalLinear
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import my_sinkhorn_ops


#parameters
n_numbers = 64
lr = 0.1
temperature = 1.0
batch_size = 10
prob_inc = 1.0
samples_per_num = 1
n_iter_sinkhorn = 10
n_units =32
noise_factor= 1.0
optimizer = 'adam'
keep_prob = 1.
n_epochs = 500

from torch.autograd import Variable

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition=148):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)

        self.embed = nn.Linear(n_condition, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        # print(class_id.dtype)
        # print('class_id', class_id.size()) # torch.Size([4, 148])
        # print(out.size()) #torch.Size([4, 128, 4, 4])
        # class_id = torch.randn(4,1)
        # print(self.embed)
        embed = self.embed(class_id)
        # print('embed', embed.size())
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        # print(beta.size())
        out = gamma * out + beta

        return out
import math

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

class Swish(nn.Module):  # Swish activation                                      
    def forward(self, x):
        return x * torch.sigmoid(x)

class PermutationEquivariant(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermutationEquivariant, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        print(x.size(),xm.size())
        xm = self.Lambda(xm) 
        x = self.Gamma(x)
        x = x - xm
        return x
from scipy.optimize import linear_sum_assignment
import numpy as np
import my_sinkhorn_ops


#parameters
n_numbers = 16
lr = 0.1
temperature = 1.0
batch_size = 10
prob_inc = 1.0
samples_per_num = 1
n_iter_sinkhorn = 10
n_units =32
noise_factor= 1.0
optimizer = 'adam'
keep_prob = 1.
n_epochs = 500

def inv_soft_pers_flattened(soft_perms_inf):
    inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
    inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

    inv_soft_perms_flat = inv_soft_perms.reshape(-1, n_numbers, n_numbers)
    return inv_soft_perms_flat
from modules import *
from fspool import *
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.ll = nn.LeakyReLU(0.1) # NN.UTILS.SPECTRAL_NORM
        #self.ll = Swish()
        self.ll = nn.ReLU()
        self.ln1 = nn.Linear(16,128)
        self.lbn1 = torch.nn.BatchNorm1d(64*16)
        self.ln2 = nn.Linear(128,16)
        self.lbn2 = torch.nn.BatchNorm1d(64*16)
        self.ln3 = nn.Linear(32*32,2048)
        self.lbn3 = torch.nn.BatchNorm1d(2048)
        self.ln4 = nn.Linear(2048,32*32*3)
        self.lbn4 = torch.nn.BatchNorm1d(1024)
  #      self.ln3 = nn.Linear(32*32,32*32*3))
   #     self.lbn3 = torch.nn.BatchNorm1d(32*32*3)
        self.mlp1 = nn.utils.spectral_norm(nn.Linear(256,512))
        self.mlp_bn1 =  torch.nn.BatchNorm1d(512)
        self.mlp2 = nn.utils.spectral_norm(nn.Linear(512, 2048))
        self.mlp_bn2 =  torch.nn.BatchNorm1d(2048)
        self.mlp3 = nn.utils.spectral_norm(nn.Linear(2048, 3072))
        self.mlp_bn3 =  torch.nn.BatchNorm1d(1024)


        self.key = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=False))
        self.W = torch.nn.Parameter(torch.randn(16,16))
        self.Ws = torch.nn.Parameter(torch.randn(16,16))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.W_bn = torch.nn.BatchNorm2d(64*16)
        self.W.requires_grad = True
        self.sg =  torch.nn.Sigmoid() 
        self.tn = torch.nn.Tanh()
        self.conv1  =  nn.Linear(192,1024)
        self.sbn1 = torch.nn.BatchNorm1d(1024)
        self.ps = torch.nn.PixelShuffle(2)
        self.ps_fst = torch.nn.PixelShuffle(4)
        self.co0 = nn.Conv2d(4*16, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.co_bn0 = torch.nn.BatchNorm2d(64*16)
        self.fspool = FSPool(1024,16)
        self.co1 = nn.Conv2d(64*16, 64*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.co_bn1 = torch.nn.BatchNorm2d(64*16)
        self.co2 = nn.Conv2d(64*4, 64*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.co_bn2 = torch.nn.BatchNorm2d(64*4)
        self.co3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.co_bn3 = torch.nn.BatchNorm2d(64)
        self.co4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.sm = nn.Softmax(dim=-2)
        self.sa1 = Self_Attn(16*16*4, 'relu')
        self.sa2 = Self_Attn(16, 'relu')
    def forward(self, z, labels, switch, temp):
 #         out = z.reshape(-1,32*32*3)
#          out = out[:,torch.randperm(32*32*3)]
          x_stack = None
          stack = None
          if switch != 2:
            x_stack = None
            x_stack_norm = None
            idx = 0
            c_num = 0
            for i in range(4):
              tmp = None
              tmp_norm = None
              for j in range(4):
                out = z[:,:,i*8:(i+1)*8,j*8:(j+1)*8].reshape(-1,192)
                out = out[:,torch.randperm(192)]
                out,_ = torch.sort(out, dim = -1)
                out = self.ll(self.sbn1(self.conv1(out)).view(-1,1024,1))
                if tmp is None:
                  tmp = out
                else:
                  tmp = torch.cat([tmp,out],dim=2)
                idx = idx + 1
              if x_stack is None:
                x_stack = tmp
              else:
                x_stack = torch.cat([x_stack,tmp],dim=2)

#          x_stack = x_stack[:,:,torch.randperm(16)]
          x_stack,_ = torch.sort(x_stack, dim = -1)
          x_stack,_ = torch.sort(x_stack, dim = -2)
  #        mat = self.mlp3(self.ll(self.mlp_bn2(self.mlp2(self.ll(self.mlp_bn1(self.mlp1(x_stack2.view(-1,1024*16)))))))).view(-1,16,16)
  #        mat = mat.reshape(-1,16,16)#self.out_features)#.long()
#          tmp = self.ll(self.key(x_stack.view(-1,1024,16,1).permute(0,3,1,2)).permute(0,2,3,1))
#          tmp = self.ll(self.W_bn(torch.matmul(tmp.view(-1,1024,16,32),self.W)))#       if switch == 0:
          tmp = torch.matmul(x_stack ,self.W)
 #         tmp,_ = torch.max(tmp, dim=-2)
       #   tmp = tmp + x_stack  
        #  x_stack2 = x_stack[:,:,torch.randperm(16)]
        #  tmp2 = self.ll(self.key(x_stack2.view(-1,1024,16,1).permute(0,3,1,2)).permute(0,2,3,1))
        #  tmp2 = torch.matmul(tmp2.view(-1,1024,16,128),self.W)#       if switch == 0:
        #  tmp2,_ = torch.max(tmp2, dim=-2)
    #      tmp = F.adaptive_max_pool2d(x_stack.view(-1,256,4,4), (1, 1)).view(-1,256)
     #     tmp = x_stack.view(-1,256)
      #    tmp = self.ll(self.mlp_bn1(self.mlp1(tmp)))
       ##   tmp = self.ll(self.mlp_bn2(self.mlp2(tmp)))
         # out = self.sg(self.mlp3(tmp)).view(-1,3,32,32)

#          print(tmp==tmp2)
#          exit()
 #         tmp = x_stack.permute(0,2,1).view(-1,16,32,32)
  #        tmp = self.ll(self.co_bn3(self.co3(tmp)))
   #       out = self.sg((self.co4(tmp)))
          residual = torch.abs(x_stack-x_stack)
     #     return out, out, self.Ws.reshape(1,16,16), residual
        #  return out, out, self.Ws.reshape(1,16,16), residual
          out = x_stack
          out = tmp
          out,_ = self.sa1(out.reshape(-1,1024,4,4))
#          out = self.ll(self.co_bn1(self.co1(out)))
          out = self.ps(out)
          out = self.ll(self.co_bn2(self.co2(out)))
          out = self.ps(out)
          out = self.ll(self.co_bn3(self.co3(out)))
          out = self.ps(out)
          out,_ = self.sa2(out)
#          out = self.ll(self.co_bn1(self.co1(out)))
          out = self.sg((self.co4(out)))
#          out = self.sg((x_stack)).view(z.size())
          return out, out, self.Ws.reshape(1,16,16), residual
          out,_ = torch.sort(x_stack, dim = -1)
          out = z.reshape(-1,32*32*3)
          out = self.ll(self.lbn1(self.ln1(out))) 
          out = self.ll(self.lbn2(self.ln2(out)))
          out = self.ll(self.lbn3(self.ln3(out)))
          out = self.sg((self.ln4(out))).view(z.size())
 #         out = self.ll((self.ln1(out)))
#          out = self.ll((self.ln2(out)))
  #        out = self.tn((self.ln3(out))).view(z.size())

          return out, out, mat, out
	
