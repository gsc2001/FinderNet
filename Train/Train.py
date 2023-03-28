import open3d as o3d 
import numpy as np 
import copy
import torch
import math
import argparse
import os
import pandas as pd
import csv
import torch.optim as optim
import matplotlib.pyplot as plt 
from torchvision import transforms
import DataParser as DP
import torch.nn as nn
from natsort import natsorted
from math import log2
import sys
import STModel as ST
from tqdm import tqdm
from scipy import ndimage

parser = argparse.ArgumentParser(description='Train Code for Spatial Transformer')
parser.add_argument('--data_path', help='Path to the dataset', default='unreal_00_new.csv' )
parser.add_argument('--base_path', help='path to parent directory of the image dataset folder' , default='/home2/sudarshan.s/UNREAL')
parser.add_argument('--batch_size' , help='Size of Batch' , default= 12 ) ## Batch size is the number of triplets or rows of the csv file 
parser.add_argument('--num_epochs' , help='Number of epochs' , default = 200 )
parser.add_argument('--image_resolution' , help='Size of image ' , default =250)
parser.add_argument('--save_path', help='base path to save the model' , default='/home2/sudarshan.s/Place-Recognition-Experiments/SpatialTransformer/STUpdate/FinderNetCodes/Train/UnrealModel')
parser.add_argument('--iters_per_ckpt' , help= 'number of iterations to save a checkpoint' , default=60)
parser.add_argument('--total_train_samples' , help= 'Total number of train samples' , default= 250 )
parser.add_argument('--total_test_samples' , help='Total number of validation/test samples' , default= 50)
parser.add_argument('--start_index' , help='enter row number of the csv to consider as start ', default=0)
parser.add_argument('--margin' , help ='margin of the triplet loss ', default= 2.75 )
parser.add_argument('--continue_train' , help =' Continue traiing from a previous checkpoint  ', default= True )
parser.add_argument('--path_to_prev_ckpt' , help =' path to the previous checkpoint only required if continue_train is true  ', default= 'UnrealModel/model_unreal_8.pt' )
parser.add_argument('--lr_change_frequency', help='Number of epochs to update the learning rate', default=10)
args = parser.parse_args()


torch.autograd.set_detect_anomaly(True)

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True ## makes the process deterministic i.e when a same input is given to a the same algortihm on the same hardware the result is the same
torch.backends.cudnn.benchmark = False ## It enables auto tune to find the best algorithm for the given hardware


device = "cuda"

print(" Total Samples for Training " + str(args.total_train_samples) , flush=True )
NumBatches = math.ceil(args.total_train_samples/args.batch_size)
NumValidationBatches = math.ceil(args.total_test_samples/args.batch_size)


# DF = pd.read_csv( args.data_path  )

# DF = DF.sample(frac=1).reset_index(drop=True)



def CreateBatchData( start_index , mode , DF ):

	if(mode == 'train'):
		final = args.total_train_samples
	if(mode == 'validation'):
		final = args.total_test_samples


	EndIndex = start_index + final #args.total_train_samples

	# AnchorSamples = DF['anchor'][start_index:EndIndex]
	# PositiveSamples = DF['positive'][start_index:EndIndex]
	# NegativeSamples = DF['negative'][start_index:EndIndex]


	AnchorSamples = DF['anchor'][start_index:EndIndex]
	PositiveSamples = DF['positive'][start_index:EndIndex]
	NegativeSamples = DF['negative'][start_index:EndIndex]


	AnchorDataSet = []
	PositiveDataSet = []
	NegativeDataSet  = []

	if( mode == 'train'):

		num_batches = NumBatches
		start_val = start_index

	if( mode =='validation'):
		num_batches = NumValidationBatches 
		start_val = start_index


	for i in range( num_batches):

		TempAnchor = []
		TempPositive = []
		TempNegative = []

		St= start_val + i*args.batch_size
		Ed = St + args.batch_size

		if( Ed >  args.total_train_samples ):
			Ed =  args.total_train_samples
			TotalSamples =  St - Ed 
		else :
			TotalSamples = args.batch_size

		AnchorBatch = AnchorSamples[St : Ed]
		PositiveBatch = PositiveSamples[St:Ed]
		NegativeBatch = NegativeSamples[St:Ed]


		for j in range( TotalSamples ):

			AnchorName = AnchorBatch[  St + j]
			AnchorName2 = AnchorName.split(".")
			AnchorName = str(AnchorName2[0]) + ".png"

			PositiveName = PositiveBatch[  St + j]
			PositiveName2 = PositiveName.split(".")
			PositiveName = str(PositiveName2[0]) + ".png"

			NegativeName = NegativeBatch[  St + j]
			NegativeName2 = NegativeName.split(".")
			NegativeName = str(NegativeName2[0]) + ".png"


			anchor_path = os.path.join(args.base_path ,AnchorName  )
			positive_path = os.path.join(args.base_path ,PositiveName )
			negative_path = os.path.join(args.base_path ,NegativeName )
			TempAnchor.append(anchor_path )
			TempPositive.append(positive_path)
			TempNegative.append(negative_path)

		AnchorDataSet.append(TempAnchor)
		PositiveDataSet.append(TempPositive)
		NegativeDataSet.append(TempNegative )


	return AnchorDataSet , PositiveDataSet , NegativeDataSet




# AnchorDataSet , PositiveDataSet , NegativeDataSet = CreateBatchData(args.start_index, 'train')

model = ST.Network()


criterion1 = nn.MSELoss() #torch.nn.TripletMarginLoss(margin=args.margin) #nn.BCELoss()
criterion2 = nn.MSELoss()

if(args.continue_train):
	model.load_state_dict(torch.load(args.path_to_prev_ckpt) )
	print(" Model Loaded " + str(args.path_to_prev_ckpt) ,  flush=True)

# model= nn.DataParallel(model)

print(torch.cuda.device_count())

model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

optimizer = optim.Adam(model.parameters(), lr= 4/(1000000000),betas=(0.9,0.999)  )
save_checkpoint = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_change_frequency, gamma=0.1)

print("Scheduler Enabled")

Ones = torch.ones( (args.batch_size,1) ).to("cuda")
Zeros = torch.zeros( (args.batch_size ,1) ).to("cuda")


Cntr =0


# def GetCorrespondanceLoss( Embd_a , Embd_p):

# 	a,b,c,d = Embd_a.size()

# 	for i in range(a):



def GetCentroidPositions( Embd ):

	a,b,c,d = Embd.size()

	Embd = Embd.to("cpu").detach().numpy()

	COM = torch.zeros( ( a,3,1 ) )

	for i in range(a):
		Q = torch.reshape( torch.tensor( ndimage.center_of_mass(Embd[i])  ) - torch.tensor([0, c/2, d/2]) , (3,1))
		COM[i] = torch.reshape(torch.tensor( [ Q[1] , Q[2] ,Q[0] ]) , (3,1))

	COM = COM.to("cuda")

	return COM




for eph in range( args.num_epochs):
	EphochLoss =0
	correct = 0
	validaton_correct = 0
	CumilativeLoss1 = 0
	CumilativeLoss2 = 0
	CumilativeLoss3 = 0
	TranslationCum = 0 

	DF = pd.read_csv( args.data_path  )
	DF = DF.sample(frac=1).reset_index(drop=True)

	AnchorDataSet , PositiveDataSet , NegativeDataSet = CreateBatchData(args.start_index, 'train' , DF)


	for batch_num in tqdm(range(NumBatches)):
		optimizer.zero_grad()

		if( len( AnchorDataSet[batch_num]) > 0 ):

			AnchorImgs = DP.ReadImages2(AnchorDataSet[batch_num] , True )
			PositiveImgs = DP.ReadImages2(PositiveDataSet[batch_num] , True )
			NegativeImgs = DP.ReadImages2(NegativeDataSet[batch_num] , True )

			AnchorImgs = AnchorImgs.unsqueeze(1)
			PositiveImgs = PositiveImgs.unsqueeze(1)
			NegativeImgs = NegativeImgs.unsqueeze(1)

			# Scores_ap , Reconstructed_DEM_a1 , Reconstructed_DEM_p = model.forward(AnchorImgs , PositiveImgs)
			# Scores_an , Reconstructed_DEM_a2 , Reconstructed_DEM_n  = model.forward(AnchorImgs , NegativeImgs)


			Embd_a1, Embd_p,  Reconstructed_DEM_a1 , Reconstructed_DEM_p  = model.forward(AnchorImgs , PositiveImgs)
			Embd_a2, Embd_n,  Reconstructed_DEM_a2 , Reconstructed_DEM_n  = model.forward(AnchorImgs , NegativeImgs)


			Scores_ap = model.DeltaLayer( Embd_a1, Embd_p )
			Scores_an = model.DeltaLayer( Embd_a2, Embd_n )

			Loss1 = torch.sum( torch.maximum( Scores_ap -  Scores_an + args.margin , Zeros).to("cuda"))  #criterion1(Scores_an, Zeros) + criterion1(Scores_ap, Ones)
			Loss2 = criterion2(Reconstructed_DEM_a1 , AnchorImgs ) +  criterion2(Reconstructed_DEM_a2 , AnchorImgs ) 
			+ criterion2(Reconstructed_DEM_p , PositiveImgs ) + criterion2(Reconstructed_DEM_n , NegativeImgs ) 

			# Loss3 = criterion2( T_AP_IMG , T_AP_EMBD

			CumilativeLoss1 += Loss1
			CumilativeLoss2 += 0.1*Loss2

			# TranslationCum += 0.1*Loss3

			Loss = Loss1 + 0.1*Loss2  
			Loss.backward()
			optimizer.step()

			EphochLoss += Loss

	Cntr += 1

	print( "Epoch Loss  =  "  + str(EphochLoss)  +  " classification = " + str(CumilativeLoss1) + " reconstruction = " + str(CumilativeLoss2) + " Epoch Number =  " + str(eph) ,  flush=True )
	if( Cntr % 3 == 0):
		save_file_name = "model_" + "epoch_num_" +str(eph) + "_Loss_" + str(EphochLoss) +  ".pt"
		save_file_name = os.path.join( args.save_path , save_file_name)
		torch.save(model.state_dict(), save_file_name)
