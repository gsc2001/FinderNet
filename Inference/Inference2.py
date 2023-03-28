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
import pickle 


parser = argparse.ArgumentParser(description='Train Code for Spatial Transformer')
parser.add_argument('--data_path', help='Path to the dataset', default='test_00.csv' )
parser.add_argument('--base_path', help='path to parent directory of the image dataset folder' , default='/home2/sudarshan.s/KITTI_DEM/')
parser.add_argument('--batch_size' , help='Size of Batch' , default= 1 ) ## Batch size is the number of triplets or rows of the csv file 
parser.add_argument('--num_epochs' , help='Number of epochs' , default = 250 )
parser.add_argument('--image_resolution' , help='Size of image ' , default =501)
parser.add_argument('--save_path', help='base path to save the model' , default='/home2/sudarshan.s/Place Recognition Experiments/Spatial Transformer/Model3/')
parser.add_argument('--iters_per_ckpt' , help= 'number of iterations to save a checkpoint' , default=60)
parser.add_argument('--total_test_samples' , help='Total number of validation/test samples' , default= 50)
parser.add_argument('--start_index' , help='Start index to extract data' , default= 0)
# parser.add_argument('--ckpt_path' , help ='path to model  ', default= '/home2/sudarshan.s/Place-Recognition-Experiments/SpatialTransformer/STUpdate/Oxford/model1.pt' )
parser.add_argument('--ckpt_path' , help ='path to model  ', default= 'Model/Kitti/model_08_3.pt' )
parser.add_argument('--total_samples_in_database', help='total number of samples in databse', default=50)
parser.add_argument('--test_sequence_id', help='sequence id to be tested', default='00')
parser.add_argument('--pose_base_dir', help='Base directory to the path of poses', default='/home2/sudarshan.s/Place-Recognition-Experiments/SpatialTransformer/STUpdate/FinderNetCodes/Inference/')
parser.add_argument('--dist_threshold_for_loop_candidate', help='Total number of loop candidates to select', default=60 )
parser.add_argument('--dont_consider_past', help='set of immedaiate past point clousd to reject', default=51)
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)


SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True ## makes the process deterministic i.e when a same input is given to a the same algortihm on the same hardware the result is the same
torch.backends.cudnn.benchmark = False ## It enables auto tune to find the best algorithm for the given hardware


device = "cuda"


print(" Total Samples for Training " + str(args.total_test_samples) , flush =True )
NumBatches = math.ceil(args.total_test_samples/args.batch_size)
NumValidationBatches = math.ceil(args.total_test_samples/args.batch_size)


DF = pd.read_csv( args.data_path  )



model = ST.Network().to("cuda")
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(torch.load(args.ckpt_path) )#.to("cuda")


print( "Loaded Model  " +  str(args.ckpt_path))

DEM_list = sorted(os.listdir( os.path.join( args.base_path , args.test_sequence_id) ) , key=lambda x : int(os.path.splitext(x)[0]) )

Poses = np.loadtxt( os.path.join( args.pose_base_dir ,  str(args.test_sequence_id) +".txt"  ) )

Pose_Val= [] 

for i in range( len(Poses)):

	Val = [ Poses[i][3] , Poses[i][7] , Poses[i][11] ]
	Pose_Val.append(Val)

Pose_Val = np.asarray(Pose_Val)
AllPoses = Pose_Val.T


def GetGTVector( index ):

	QPose =  np.reshape( Pose_Val[index].T , (3,1) ) 
	Distances =  np.linalg.norm( AllPoses - QPose, axis= 0 )

	Dist = Distances[ 0: index-args.dont_consider_past ]

	Loops = np.where( Dist <4  , 1, 0 )

	if 1 in Loops:
		# print( str(index) + " Loop Detected ")
		Loops2.append(1)

	return Loops 


def LoopCandidates(index):

	QPose =  np.reshape( Pose_Val[index].T , (3,1) ) 
	Distances =  np.linalg.norm( AllPoses - QPose, axis= 0 )

	Dist = Distances[ 0: index-args.dont_consider_past ]

	Loop_candidates = np.where( Dist < args.dist_threshold_for_loop_candidate  , 1, 0 )

	candidate_index = []

	GTLoops = [] 

	for i in range(len(Loop_candidates)):

		if( Loop_candidates[i] ==1 ):

			candidate_index.append(i)

			if( Dist[i] < 3 ):

				GTLoops.append(1)
			else:

				GTLoops.append(0)


	return candidate_index, GTLoops



AllGTloops = []
Detections_Distance = []


Cntr = 0 

for i in range( args.dont_consider_past+1, len(DEM_list) , 10 ):

	ImgPath =  os.path.join( os.path.join(args.base_path , args.test_sequence_id) , DEM_list[i] )
	QueryImage = DP.ReadImages2( [ImgPath] , True  )
	QueryImage = QueryImage.unsqueeze(1)


	Candidate_index , GTLoops = LoopCandidates(i)

	if(len(Candidate_index) > 0 ):

		Cntr += 1

		print(Cntr , len(Candidate_index))

		AllGTloops.append(GTLoops)
		Distance_per_query = []

		for k in (Candidate_index):

			TestImgPath = os.path.join( os.path.join(args.base_path , args.test_sequence_id) , DEM_list[k] )

			TestImg = DP.ReadImages2( [TestImgPath] , True  )
			TestImg = TestImg.unsqueeze(1)

			Embd_a1, Embd_p,  Reconstructed_DEM_a1 , Reconstructed_DEM_p   = model.forward(QueryImage , TestImg )

			Distance = model.DeltaLayer( Embd_a1, Embd_p)

			Distance_per_query.append(Distance.detach().to("cpu").numpy()[0][0])

		Detections_Distance.append(Distance_per_query)




GT = []
Obtained = []

AllGTloops2 = AllGTloops
Detections_Distance2 = Detections_Distance


for i in range( len( AllGTloops) ):

	for j in range( len( AllGTloops[i] )):

		GT.append(AllGTloops[i][j]) 
		Obtained.append(-Detections_Distance[i][j])

		# if( AllGTloops[i][j] == 1 ):

		# 	print( Detections_Distance[i][j] )

precision, recall, thresholds = precision_recall_curve( GT, Obtained)

PR_DATA = {}

PR_DATA["precision"] = np.asarray(precision)
PR_DATA["recall"] = np.asarray(recall)
PR_DATA["thresholds"] = np.asarray(thresholds)

# with open('lidar_urbanscene.pickle', 'wb') as handle:
#     pickle.dump(PR_DATA, handle)

display = PrecisionRecallDisplay.from_predictions(GT, Obtained, name=" ")
_ = display.ax_.set_title("Precision-Recall curve")
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

