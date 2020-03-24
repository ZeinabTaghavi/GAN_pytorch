import torch , torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
from torchvision import datasets , transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import D,G # generator and discriminator
import matplotlib.pyplot as plt
import numpy as np

batchSize = 16
imageSize = 64

transform1 = transforms.Compose([transforms.Resize((imageSize,imageSize)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
dataset = datasets.ImageFolder(root='./data', transform=transform1)

dataloader = DataLoader(dataset,batch_size = 16 , shuffle= True)


def weight_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0,0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0,0.02)
    m.bias.data.fill_(0)




model_gen = G()
model_gen.apply(weight_init)

model_dis = D()
model_dis.apply(weight_init)




criterion = nn.BCELoss()
optimG = torch.optim.Adam(model_gen.parameters(), lr = 0.0002)
optimD = torch.optim.Adam(model_dis.parameters(), lr = 0.0002)
epochs = 1

model_dis.load_state_dict(torch.load('model_dis.pth'))
model_gen.load_state_dict(torch.load('model_gwn.pth'))
model_dis.eval()
model_gen.eval()



for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):      
        model_dis.zero_grad()
        real, _ = data
        input = real
        target = torch.ones(input.size()[0])
        output = model_dis(input)
        err_dis_real = criterion(output, target)

        noise = torch.randn(input.size()[0], 100, 1, 1)
        fake = model_gen(noise)
        target = torch.zeros(input.size()[0])
        output = model_dis(fake.detach())

        err_dis_fake = criterion(output , target)
        err_dis = err_dis_fake + err_dis_real
        err_dis.backward()
        optimD.step()


        model_gen.zero_grad()
        target = torch.ones(input.size()[0])
        output = model_dis(fake)
        err_gen = criterion(output , target)
        err_gen.backward()
        optimG.step()

        print(epoch , i)
        if i % 100 == 0 :
          vutils.save_image(real , f'real_image{i}.jpg',normalize=True)
          vutils.save_image(fake.data , f'fake_image{i}.jpg',normalize=True)

        

torch.save(model_dis.state_dict(), 'model_dis.pth')
torch.save(model_gen.state_dict(), 'model_gwn.pth')
