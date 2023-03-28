import os
import numpy as np
import math
from PIL import Image
import PIL
from types import SimpleNamespace
from functools import partial

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
## Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
## PyTorch Lightning
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.nn.functional as F
import cv2
from scipy import ndimage


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # x = F.relu(self.t_conv1(x))
        # x = F.relu(self.t_conv2(x))

        return x




class Decoder(nn.Module):
    def __init__(self):
        super( Decoder, self).__init__()

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        return x

class CanDecoder(nn.Module):
    def __init__(self):
        super( CanDecoder, self).__init__()

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        return x

class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=4 * 125 * 125,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(0.8),
            nn.Linear(in_features=20, out_features=1),
            # nn.Tanh(),
            nn.Sigmoid()
        )
        # bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        # nn.init.constant_(self.fc[3].weight, 0)
        # self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1))#.view(batch_size, 2, 3)
        theta = 2*theta*np.pi 

        Mat = torch.zeros(( batch_size , 2,3) ).to("cuda")
        mat = torch.zeros( ( 2,2) ).to("cuda")


        for i in range(batch_size):
            mat = torch.tensor( [ torch.cos(theta[i]) , -torch.sin(theta[i]) , torch.sin(theta[i]) , torch.cos(theta[i]) ]  ).view( 2,2)
            Mat[i, 0:2 , 0:2] = mat

        grid = F.affine_grid(Mat, torch.Size((batch_size, 4 , 125 , 125)) , align_corners=True)
        img_transform = F.grid_sample(img, grid , align_corners=True ) 

        return img_transform , Mat


class ConvolutionLayers( nn.Module):

	def __init__(self):
		super(ConvolutionLayers, self).__init__()

		self.conv3 = nn.Conv2d(4, 16, 3, padding=1)
		self.conv4 = nn.Conv2d(16, 64, 5, padding=1)
		self.pool = nn.MaxPool2d(2, 2)

		self.fc2 = nn.Sequential(
            nn.Linear(in_features=64 * 30 * 30,
                      out_features=512),
            nn.Dropout(0.8),
            nn.Linear(in_features=512, out_features=128),
        )


	def forward( self, x ):

			x = F.relu(self.conv3(x))
			x = self.pool(x)
			x = F.relu(self.conv4(x))
			x = self.pool(x)
			x = torch.flatten(x, start_dim=1)
			x = self.fc2(x)

			return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                # nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input)) #nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.conv2(input)) #nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                # nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        # self.bn1 = nn.BatchNorm2d(out_channels//4)
        # self.bn2 = nn.BatchNorm2d(out_channels//4)
        # self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input)) #nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.conv2(input)) #nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.conv3(input)) #nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 10, 10, 128, 128]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)


    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)

        # input = self.layer2(input)
        # input = self.layer3(input)
        # input = self.layer4(input)
        # input = self.gap(input)
        #torch.flatten()
        #https://stackoverflow.com/questions/60115633/pytorch-flatten-doesnt-maintain-batch-size
        # input = torch.flatten(input, start_dim=1)
        # input = self.fc(input)

        return input


class DeepNetwork( nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()

        self.conv_1 = nn.Conv2d(10, 64, 5, padding=0, stride=2)
        self.conv_2 = nn.Conv2d(64, 32, 5, padding=0, stride=2)
        self.conv_3 = nn.Conv2d(32 , 4 , 1 , padding=0, stride=2) 
        self.fc1 = torch.nn.Linear(64516 , 100) #torch.nn.Linear(640, 100) 
        self.fc2 = torch.nn.Linear(100, 10) 
        self.fc3 = torch.nn.Linear(10, 1) 
        self.sig = nn.Sigmoid()

    def forward(self , x ):

        x =  F.relu(self.conv_1(x))
        # x = self.pool1(x)
        x = F.relu(self.conv_2(x))
        # x = self.pool2(x)
        x = F.relu( self.conv_3(x) )

        x = torch.flatten(x, start_dim=1)
        # x =  self.sig(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.abs( self.fc3(x) ) 

        return x 


class MutualPolarTransformer(nn.Module):

    def __init__(self):
        super(MutualPolarTransformer, self).__init__()

        self.fc1 = torch.nn.Linear(126, 64).to("cuda")
        self.fc2 = torch.nn.Linear(64, 1).to("cuda")
        self.sig = nn.Sigmoid()


    def polar_grid(self, output_size, ulim=(0, np.sqrt(2.) ), vlim=(-np.pi, np.pi), out=None, device=None):
        """Polar coordinate system.
        
        Args:
            output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
            ulim: (float, float), radial coordinate limits
            vlim: (float, float), angular coordinate limits
            out: torch.FloatTensor, output tensor
            device: string or torch.device, device for torch.tensor
            
        Returns:
            torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
            (x, y)-coordinates of the grid point.
        """
        nv, nu = output_size
        urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
        vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
        vs, us = torch.meshgrid([vrange, urange])
        xs = us * torch.cos(vs) 
        ys = us * torch.sin(vs) 

        return torch.stack([xs, ys], 2, out=out)


    def PlotPolar(self, input1, input2 , Agumented ):

        if( not Agumented ):

            fig = plt.figure()
            A1 = input1.detach().to("cpu").numpy()
            A1 = A1[0][0]

            A2 = input2.detach().to("cpu").numpy()
            A2 = A2[0][0]

            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)

            ax1.title.set_text("Polar Form of Anchor ")
            ax2.title.set_text("Polar Form of Positive ")

            ax1.imshow( A1 )
            ax2.imshow( A2 )

            plt.show()

        if(Agumented):

            A1 = input1.detach().to("cpu").numpy()
            A1 = A1[0][0]

            plt.imshow(A1)
            plt.show()


    def PlotSoftMax( self, softmax_result ):

        softmax_result = softmax_result.detach().to("cpu").numpy()
        # print(softmax_result)

        Angle = np.linspace( -180 , 180, 126 )
        softmax_result = np.zeros(126)
        softmax_result[78] =1 

        plt.plot(Angle , softmax_result )
        plt.xlabel(" Yaw Angles ")
        plt.ylabel("Predicted Probablity")
        plt.show()







    def Correlation(self, input1, input2 , Embd1 ):

        ## Input 1 and Input 2 are in polar coordinates 


        # Translated = torch.flip( input1, [2] )
        Translated = torch.flip( input1, [1,0] )
        
        # self.PlotPolar( input1, Translated, False)

        # print( torch.linalg.norm( Translated - input1))

        self.AugumentedFeatureVolume = torch.cat( (input1 , Translated), dim=2 ).to("cuda")
        # self.AugumentedFeatureVolume2 = torch.cat( (input1 , input1), dim=2 ).to("cuda")

        # self.PlotPolar( self.AugumentedFeatureVolume, input2, True)
        # self.PlotPolar( self.AugumentedFeatureVolume2, input2, True)


        [a,b,c,d] = input1.size() 
        self.FinalLayer = torch.zeros( ( a, 126 )).to("cuda")
        PredefinedAngles = torch.linspace( -np.pi , np.pi, 126 ).to("cuda")
        theta = torch.zeros( (a , 1)).to("cuda")

        for j in range(a):

            x = self.AugumentedFeatureVolume[j ,:,:,:]
            x = x.unsqueeze(0)
            w = input2[j , : , : , :]

            w = w.unsqueeze(0)
            x = torch.nn.functional.conv2d( input=x, weight= w )
            Temp = nn.functional.softmax( x[0, 0,:,0] )
            # self.PlotSoftMax(Temp)
            theta[j ,0] =   torch.sum(torch.mul(Temp, PredefinedAngles)) - np.pi #x[0, 0,:,0]

            # print( theta[j])


        Mat = torch.zeros(( a , 2,3) ).to("cuda")
        mat = torch.zeros( ( 2,2) ).to("cuda")


        for i in range(a):
            mat = torch.tensor( [ torch.cos(theta[i]) , -torch.sin(theta[i]) , torch.sin(theta[i]) , torch.cos(theta[i]) ]  ).view( 2,2)
            Mat[i, 0:2 , 0:2] = mat

        grid = F.affine_grid(Mat, torch.Size((a, 4 , 125 , 125)) , align_corners=True)
        img_transform = F.grid_sample(Embd1, grid , align_corners=True ) 

        return img_transform, Mat



    def forward(self, Embd1, Embd2 ):

        a,b,c,d = Embd1.size()

        grid = torch.zeros(( a , 125 ,125 , 2) ).to("cuda")

        for i in range(a):
            grid[i] = self.polar_grid( (125,125)  )


        Embd1_Polar = F.grid_sample(Embd1, grid , align_corners=True )
        Embd2_Polar = F.grid_sample(Embd2, grid , align_corners=True ) 

        TransformedEmbd1, R_Mat = self.Correlation( Embd1_Polar , Embd2_Polar , Embd1)

        return TransformedEmbd1 , R_Mat



class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.ecnoder = ConvEncoder()
        self.ST = LocalNetwork()
        self.Dec = Decoder()
        self.Conv = ResNet(4, ResBottleneckBlock, [2, 2, 2, 2], useBottleneck=False, outputs=256)  #ConvolutionLayers()

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256,
                      out_features=1))
        self.sigmoid_function = nn.Sigmoid()
        self.DNN = DeepNetwork()
        self.Dec2 = CanDecoder()

        self.MPT = MutualPolarTransformer()

        self.bn1 = nn.BatchNorm2d(10)


    def PlotImages( self, x1,Reconstructed_DEM ,Transformed_Embd1 , x2):

        Original = x1.to("cpu").detach().numpy()[0][0]
        Positive = x2.to("cpu").detach().numpy()[0][0]
        Reconstructed = Reconstructed_DEM.to("cpu").detach().numpy()[0][0]
        # row,col = Reconstructed.shape #int(theta[0][0]*180/3.14)
        # center=tuple(np.array([row,col])/2)
        # rot_mat = cv2.getRotationMatrix2D(center,int(theta[0][0]*180/3.14),1.0)



        Transformed_Embd1_1 = Transformed_Embd1.to("cpu").detach().numpy()[0][0]
        Transformed_Embd2 = Transformed_Embd1.to("cpu").detach().numpy()[0][1]
        Transformed_Embd3 = Transformed_Embd1.to("cpu").detach().numpy()[0][2]
        Transformed_Embd4 = Transformed_Embd1.to("cpu").detach().numpy()[0][3]

        # Canonicalized = cv2.warpAffine(Reconstructed, rot_mat, (col,row))

        fig = plt.figure()

        a,b = np.shape(Original)

        O = np.zeros( ( a,b,3 ) )
        R = np.zeros( ( a,b,3 ) )
        P = np.zeros( (a,b,3))

        O = np.dstack(( Original , Original , Original))
        R = np.dstack(( Reconstructed , Reconstructed , Reconstructed))
        P = np.dstack((Positive , Positive, Positive))

        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")

        ax1.imshow( O )
        # ax1.title.set_text("Original DEM ")


        ax2.imshow( R )
        # ax2.title.set_text("Canonicalized DEM ")


        ax3.imshow( P )
        # ax3.title.set_text("Positive DEM ")

        # ax1 = fig.add_subplot(2,3,1)
        # ax2 = fig.add_subplot(2,3,2)
        # ax3 = fig.add_subplot(2,3,3)
        # ax4 = fig.add_subplot(2,3,4)
        # ax5 = fig.add_subplot(2,3,5)
        # ax6 = fig.add_subplot(2,3,6)

        # ax1.imshow( Original )
        # ax1.title.set_text("Original DEM ")
        # ax2.imshow(Reconstructed)
        # ax2.title.set_text("Reconstructed DEM ")
        # ax3.imshow(Transformed_Embd1_1 )
        # ax3.title.set_text("Canonicalized DEM 0 ")
        # ax4.imshow(Transformed_Embd2 )
        # ax4.title.set_text("Positive DEM ")
        # ax5.imshow(Positive )
        # ax5.title.set_text("Canonicalized DEM 2 ")
        # ax6.imshow(Transformed_Embd4 )
        # ax6.title.set_text("Canonicalized DEM 3 ")
        plt.show()


    def forward(self, x1 , x2 ):

        CompressedEmbd1 = self.ecnoder(x1)
        CompressedEmbd2 = self.ecnoder(x2)

        Reconstructed_DEM1 = self.Dec(CompressedEmbd1)
        Reconstructed_DEM2 = self.Dec(CompressedEmbd2)

        Transformed_Embd1, R_Mat = self.MPT.forward(CompressedEmbd1,  CompressedEmbd2)

        Embd1 = self.Conv(Transformed_Embd1) ## Anchor would be transformed
        Embd2 = self.Conv(CompressedEmbd2)

        # if(Loop):
        # print( Embd1.size()) 
        # self.PlotImages( CompressedEmbd1, CompressedEmbd1, Transformed_Embd1 ,CompressedEmbd2)


        return Embd1, Embd2,  Reconstructed_DEM1 , Reconstructed_DEM2

    def DeltaLayer( self, Embd1, Embd2 ):

        a,b,c,d = Embd1.size()

        A1 = torch.reshape(Embd1 , ( a,b,c*d )  )
        A2 = torch.reshape(Embd2 , ( a,b,c*d )  )

        A1 = A1.unsqueeze(2)
        A2 = A2.unsqueeze(2)

        T_A1 = torch.tile(A1, (c*d , 1 ) )
        T_A2 = torch.transpose( torch.tile( A2, (c*d , 1) ) , 2,3) 

        Diff = torch.abs(T_A1 - T_A2)

        # Diff= self.bn1(Diff)

        # print(Diff.size() )
        Distance = self.DNN(Diff)

        return Distance


