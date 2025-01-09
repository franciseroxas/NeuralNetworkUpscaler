import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from PIL import Image
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class DandadanImageDataset(Dataset):
    def __init__(self, small_img_dir, big_img_dir):
        self.small_img_dir = small_img_dir
        self.big_img_dir = big_img_dir

    def __len__(self):
        return len(os.listdir(self.small_img_dir))

    def __getitem__(self, idx):
        small_img_path = self.small_img_dir + os.listdir(self.small_img_dir)[idx]
        big_img_path = self.big_img_dir + os.listdir(self.big_img_dir)[idx]
        
        #Load in BGR 
        smallImage = torch.Tensor(cv2.imread(small_img_path, cv2.IMREAD_GRAYSCALE)) / 255
        bigImage = torch.Tensor(cv2.imread(big_img_path, cv2.IMREAD_GRAYSCALE)) / 255

        return smallImage.unsqueeze(0), bigImage.unsqueeze(0) #feature dim is on the first dimension after batch size

#Change to a better upscaler after getting this one to work
class UNetUpscaler(nn.Module):

    def __init__(self):
        super(UNetUpscaler, self).__init__()
        #Self feature transform for edges
        self.edgeTransform1 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                           nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3,3)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 32),
                                           nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 32))
        
        self.edgeTransform2 = nn.Sequential(nn.ConstantPad2d(2, -1),
                                           nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3,3), dilation = (2,2)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 32),
                                           nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 32))
        
        #Encoder
        self.encode1 = nn.Sequential(nn.Conv2d(in_channels = 65, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                             
        self.encode2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                                                          
        self.encode3 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                             
        #BottleNeck
        self.bottleNeck1 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                     
        self.bottleNeck2 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
        
        #Decoder
        self.decode1 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                     
        self.decode2 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64),
                                     nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                     
        self.decode3 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), dilation = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
                                     
        #CNN for learning superresolution 
        self.superResolution1 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                              nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 64),
                                              nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 64),
                                              nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 256))
                                     
        self.superResolution2 = nn.Sequential(nn.ConstantPad2d(1, -1), 
                                              nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3,3)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 32),
                                              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 32),
                                              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 32))
        
        self.superResolution3 = nn.Sequential(nn.ConstantPad2d(2, -1), 
                                              nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3,3), dilation = (2,2)),
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 32),
                                              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 32),
                                              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)), 
                                              nn.ReLU(), 
                                              nn.BatchNorm2d(num_features = 32))
                                     
        self.convolutionalizationLayers = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (1,1)),
                                                        nn.ReLU(), 
                                                        nn.BatchNorm2d(num_features = 64),
                                                        nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (1,1)),
                                                        nn.ReLU(), 
                                                        nn.BatchNorm2d(num_features = 32),
                                                        nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = (1,1)))
                                                        
    def forward(self, input):
        #Make sure that the edge effects are dealt with
        encoderInput1 = self.edgeTransform1(input)
        encoderInput2 = self.edgeTransform2(input)
        
        encode1 = self.encode1(torch.concatenate((encoderInput1, encoderInput2, input), 1))
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        
        bottleNeck1 = self.bottleNeck1(encode3)
        bottleNeck2 = self.bottleNeck2(bottleNeck1)
        
        decode1 = self.decode1(torch.concatenate((bottleNeck2, encode3), 1))
        decode2 = self.decode2(torch.concatenate((decode1, encode2), 1))
        decode3 = self.decode3(torch.concatenate((decode2, encode1), 1))
        
        #upscale by 2 times using channel dimension
        outputHighScale1 = torch.zeros(decode3.shape[0], int(decode3.shape[1]/4), decode3.shape[2]*2, decode3.shape[3]*2, device = decode3.device)
        outputHighScale1[:, :, 0::2, 0::2] = decode3[:, 0::4, :, :]
        outputHighScale1[:, :, 1::2, 0::2] = decode3[:, 1::4, :, :]
        outputHighScale1[:, :, 0::2, 1::2] = decode3[:, 2::4, :, :]
        outputHighScale1[:, :, 1::2, 1::2] = decode3[:, 3::4, :, :]
        
        outputHighScale1 = self.superResolution1(outputHighScale1)
        
        #upscale by 2 times using channel dimension
        outputHighScale2 = torch.zeros(outputHighScale1.shape[0], int(outputHighScale1.shape[1]/4), outputHighScale1.shape[2]*2, outputHighScale1.shape[3]*2, device = outputHighScale1.device)
        outputHighScale2[:, :, 0::2, 0::2] = outputHighScale1[:, 0::4, :, :]
        outputHighScale2[:, :, 1::2, 0::2] = outputHighScale1[:, 1::4, :, :]
        outputHighScale2[:, :, 0::2, 1::2] = outputHighScale1[:, 2::4, :, :]
        outputHighScale2[:, :, 1::2, 1::2] = outputHighScale1[:, 3::4, :, :]
                
        superResolution2 = self.superResolution2(outputHighScale2)
        superResolution3 = self.superResolution3(outputHighScale2)
        
        output = self.convolutionalizationLayers(torch.concatenate((superResolution2, superResolution3, outputHighScale2), 1))
        
        return output

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        return (torch.mean((predictions - targets) ** 2) + torch.nn.functional.huber_loss(predictions, targets))/2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
if(os.path.exists("./datasets.pt")):
    datasetsDict = torch.load("./datasets.pt")
    trainDataset = datasetsDict['trainDataset']
    valDataset = datasetsDict['valDataset']
    del datasetsDict
else:
    myDataset = DandadanImageDataset(small_img_dir = "dataset_small2/", big_img_dir = "dataset_big2/")
    trainDataset, valDataset = random_split(myDataset, [0.8, 0.2])
    torch.save({'trainDataset': trainDataset,
                'valDataset': valDataset}, "./datasets.pt")

batch_size = 8
trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = UNetUpscaler()
if torch.cuda.is_available():
    model.cuda()
    
learningRate = 0.0001/batch_size
num_epochs = 100

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
lossOverTime = []

for epoch in range(num_epochs):
    start = time.time()
    currentLoss = 0
    for batch_ndx, data in enumerate(trainDataloader):
        smallImgsTensor, bigImgsTensor = data
        smallImgsTensor, bigImgsTensor = smallImgsTensor.to(device), bigImgsTensor.to(device)
        
        upscaledOutput = model(smallImgsTensor)
        loss = criterion(bigImgsTensor, upscaledOutput) #loss is symmetric 
        batchLoss = loss.cpu().detach().numpy() / batch_size
        currentLoss = currentLoss + batchLoss #normalize by batch size for comparison
        print("Epoch", epoch, " | Batch", batch_ndx, " | Total", len(trainDataloader), " | LR:", scheduler.get_lr()[0], " | Batch Loss:", batchLoss,)
        
        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #For visualization of an output
        cv2.imwrite("output/test" + str(batch_ndx % 50) + "_upscaled.png", upscaledOutput[0][0].cpu().detach().numpy() * 255)
        cv2.imwrite("output/test" + str(batch_ndx % 50) + "_actual.png", smallImgsTensor[0][0].cpu().detach().numpy() * 255)
        
        if(batch_ndx % 10 == 0):
            try:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': loss,
                        'batch_ndx': batch_ndx,
                        'learningRate': learningRate,
                        'batch_size': batch_size,
                        }, "./upscalingModel_inProgress.pt")
            except Exception as e:
                print(e)

    scheduler.step()
    time.sleep(1)
    print("----------------------------")
    print("Epoch", epoch, "Loss:", currentLoss, "Time:", time.time() - start)
    print("----------------------------")
        
    lossOverTime.append(currentLoss)
    plt.figure()
    plt.plot(np.array(lossOverTime))
    plt.title("Loss of neural network over time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Loss of neural network.png")
    plt.close("all")

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'batch_ndx': batch_ndx,
            'learningRate': learningRate,
            'batch_size': batch_size,
            }, "./upscalingModel_inProgress.pt")
