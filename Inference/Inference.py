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
from PIL import Image
import PIL
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import time
import json

parser = argparse.ArgumentParser(description='Train Code for Spatial Transformer')
parser.add_argument('--data_path', help='Path to the dataset', default='test_00.csv' )
parser.add_argument('--base_path', help='path to parent directory of the image dataset folder' , default='/home2/sudarshan.s/PandaDEM')
parser.add_argument('--batch_size' , help='Size of Batch' , default= 1 ) ## Batch size is the number of triplets or rows of the csv file 
# parser.add_argument('--num_epochs' , help='Number of epochs' , default = 250 )
parser.add_argument('--image_resolution' , help='Size of image ' , default =501)
# parser.add_argument('--save_path', help='base path to save the model' , default='/home2/sudarshan.s/Place Recognition Experiments/Spatial Transformer/Model3/')
parser.add_argument('--iters_per_ckpt' , help= 'number of iterations to save a checkpoint' , default=60)
# parser.add_argument('--total_test_samples' , help='Total number of validation/test samples' , default= 50)
parser.add_argument('--start_index' , help='Start index to extract data' , default= 0)
parser.add_argument('--ckpt_path' , help ='path to model  ', default= 'Model/Panda/model1.pt' )
# parser.add_argument('--total_samples_in_database', help='total number of samples in databse', default=50)
parser.add_argument('--test_sequence_id', help='sequence id to be tested', default='124')
parser.add_argument('--pose_path', help='Base directory to the path of poses', default='/home2/sudarshan.s/PandaPoses/')
parser.add_argument('--ignore_past', help='Ignore the past point clouds' , default=1)
parser.add_argument('--sample_frequency', help='Ignore the past point clouds' , default=2)
parser.add_argument('--GT_positive_distance', help="distnace to consider a smaple as positive", default=4)
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)


args.pose_path = os.path.join( args.pose_path , args.test_sequence_id+".json" )

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True ## makes the process deterministic i.e when a same input is given to a the same algortihm on the same hardware the result is the same
torch.backends.cudnn.benchmark = False ## It enables auto tune to find the best algorithm for the given hardware


device = "cuda"


# print(" Total Samples for Training " + str(args.total_test_samples) , flush =True )
# NumBatches = math.ceil(args.total_test_samples/args.batch_size)
# NumValidationBatches = math.ceil(args.total_test_samples/args.batch_size)


DF = pd.read_csv( args.data_path  )



model = ST.Network().to("cuda")
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(torch.load(args.ckpt_path) )#.to("cuda")


print( "Loaded Model  " +  str(args.ckpt_path))

DEM_list = sorted(os.listdir( os.path.join( args.base_path , args.test_sequence_id) ) , key=lambda x : int(os.path.splitext(x)[0]) )


f = open(args.pose_path)

out = json.load(f)
NumPoses = len(out)
Poses = np.zeros( (NumPoses , 3 ))

for i in range(NumPoses):
	
	X = out[i]["position"]["x"]
	Y = out[i]["position"]["y"]
	Z = out[i]["position"]["z"] 
	Poses[i] = [ X, Y, Z]

Pose_Val = np.asarray(Poses)
AllPoses = Pose_Val.T

print( Pose_Val[0] , Pose_Val[-1])

Loops2  = []

def GetGTVector( index ):

	QPose =  np.reshape( Pose_Val[index].T , (3,1) ) 
	Distances =  np.linalg.norm( AllPoses - QPose, axis= 0 )
	# print(Distances)

	Dist = Distances[ 0: index-args.ignore_past ]
	# print(Dist)

	Loops = np.where( Dist <args.GT_positive_distance  , 1, 0 )

	if 1 in Loops:
		# print( str(index) + " Loop Detected ")
		Loops2.append(1)

	return Loops 


AllGTloops = []
Detections_Distance = []

for i in range( args.ignore_past, len(DEM_list) , args.sample_frequency ):

	ImgPath =  os.path.join( os.path.join(args.base_path , args.test_sequence_id) , DEM_list[i] )

	QueryImage = DP.ReadImages2( [ImgPath] , True  )
	QueryImage = QueryImage.unsqueeze(1)
	GTLoops  = GetGTVector(i)
	AllGTloops.append(GTLoops)

	Distance_per_query = []

	print("checking for " + str(i) )

	for j in range( 0, (i-args.ignore_past) ):

		# t1 = time.time()

		TestImgPath = os.path.join( os.path.join(args.base_path , args.test_sequence_id) , DEM_list[j] )
		TestImg = DP.ReadImages2( [TestImgPath] , True  )
		TestImg = TestImg.unsqueeze(1)

		Embd_a1, Embd_p,  Reconstructed_DEM_a1 , Reconstructed_DEM_p   = model.forward(QueryImage , TestImg )

		Distance = model.DeltaLayer( Embd_a1, Embd_p)

		t2 = time.time()

		# print( t2 - t1)
		# print(Distance)


		Distance_per_query.append(Distance.detach().to("cpu").numpy()[0][0])

	res = sorted(range(len(Distance_per_query)), key = lambda sub: Distance_per_query[sub])[:3]

	print( str(i) + " " + str(res))

	Detections_Distance.append(Distance_per_query)



GT = []
Obtained = []

AllGTloops2 = AllGTloops
Detections_Distance2 = Detections_Distance


for i in range( len( AllGTloops) ):

	for j in range( len( AllGTloops[i] )):

		GT.append(AllGTloops[i][j]) 
		Obtained.append(-Detections_Distance[i][j])

		if( AllGTloops[i][j] == 1 ):

			print( Detections_Distance[i][j] )

precision, recall, thresholds = precision_recall_curve( GT, Obtained)

display = PrecisionRecallDisplay.from_predictions(GT, Obtained, name=" ")
_ = display.ax_.set_title("PR curve for " + args.test_sequence_id + " sequence")
plt.show()

GT = []
Obtained = []


# for i in range( len( AllGTloops2) ):

# 		GT.append( np.amax(AllGTloops2[i]) ) 
# 		Obtained.append( -np.amin( Detections_Distance2[i] ) )

# 		if( AllGTloops2[i][j] == 1 ):

# 			print( Detections_Distance2[i][j] )


# precision, recall, thresholds = precision_recall_curve( GT, Obtained)

# display = PrecisionRecallDisplay.from_predictions(GT, Obtained, name=" ")
# _ = display.ax_.set_title("recision-Recall curve")
# plt.show()
