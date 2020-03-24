import torch , torch.nn as nn
import torch.nn.parallel

class G(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
        nn.Tanh()
    )

  def forward(self,x):
    output = self.main(x)
    return output

class D(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
          nn.Conv2d(3, 64, 4, 2, 1, bias = False),
          nn.LeakyReLU(0.2, inplace = True),
          nn.Conv2d(64, 128, 4, 2, 1, bias = False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2, inplace = True),
          nn.Conv2d(128, 256, 4, 2, 1, bias = False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.2, inplace = True),
          nn.Conv2d(256, 512, 4, 2, 1, bias = False),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2, inplace = True),
          nn.Conv2d(512, 1, 4, 1, 0, bias = False),
          nn.Sigmoid()
        )

  def forward(self,x):
    output = self.main(x)
    return output.view(-1)
