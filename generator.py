from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.ll = nn.LeakyReLU(0.1) # NN.UTILS.SPECTRAL_NORM
        self.convs0 =  nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(3, 16*16, kernel_size=4, stride=4, padding=0, bias=False)) for _ in range(64)])

        self.bns0 = nn.ModuleList([torch.nn.BatchNorm2d(16*16) for _ in range(64)])

        self.channel_opt0 = nn.utils.spectral_norm(nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn0 = torch.nn.BatchNorm2d(64)

        self.channel_opt1 = nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.channel_opt2 = nn.utils.spectral_norm(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False))

        self.rep1 = nn.Sequential(
                      nn.utils.spectral_norm(nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)),
                      torch.nn.BatchNorm2d(16*4),
                      nn.LeakyReLU(0.1),
                      nn.utils.spectral_norm(nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)),
                      torch.nn.BatchNorm2d(16*4),
                      )
        self.rep2 = nn.Sequential(
                      nn.utils.spectral_norm(nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)),
                      torch.nn.BatchNorm2d(16*4),
                      nn.LeakyReLU(0.1),
                      nn.utils.spectral_norm(nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)),
                      torch.nn.BatchNorm2d(16*4),
                      )
        self.rep3 = nn.Sequential(
                      nn.utils.spectral_norm(nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)),
                      torch.nn.BatchNorm2d(16*4),
                      nn.LeakyReLU(0.1),
                      nn.utils.spectral_norm(nn.Conv2d(16*4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)),
                      torch.nn.BatchNorm2d(16*4),
                      )

        self.ps = torch.nn.PixelShuffle(4)
        self.tn = torch.nn.Tanh()

    def forward(self, z, labels):
          x_stack = None
          idx = 0
          c_num = 0
          for i in range(8):
            tmp = None
            for j in range(8):
              out = self.ll(self.bns0[idx](self.convs0[idx](z[:,:,i*4:(i+1)*4,j*4:(j+1)*4]))).view(-1,256,1)
              if tmp is None:
                tmp = out
              else:
                tmp = torch.cat([tmp,out],dim=2)
              idx = idx + 1
            if x_stack is None:
              x_stack = tmp
            else:
              x_stack = torch.cat([x_stack,tmp],dim=2)
          stack = x_stack
          x_stack = None
          for i in range(8):
            tmp = None
            for j in range(8):
              out = stack[:,:,i*8+j].contiguous().view(-1,256,1,1)
              if tmp is None:
                tmp = out
              else:
                tmp = torch.cat([tmp,out],dim=3)
            if x_stack is None:
              x_stack = tmp
            else:
              x_stack = torch.cat([x_stack,tmp],dim=2)
          feature = self.ps(x_stack)
          x_stack = feature
          x_stack = self.ll(self.bn0(self.channel_opt0(x_stack)))
          x_stack = self.ll(self.rep1(x_stack) + x_stack)
          x_stack = self.ll(self.rep2(x_stack) + x_stack)
          x_stack = self.ll(self.rep3(x_stack) + x_stack)
          x_stack2 = self.ll(self.bn1(self.channel_opt1(x_stack)))
          x_stack2 = self.tn(self.channel_opt2(x_stack2))
          return x_stack2, feature
