import numpy as np
import pandas as pd 
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2

SeqNum = "00"

parser = argparse.ArgumentParser(description='Create Train List of Triplets for the Kitti dataset')
parser.add_argument('--poses_file', help='path to the file of point clouds' , default='/home/sudarshan/Downloads/correctedPoses/denseLidar8/' + SeqNum + '.txt')
parser.add_argument('--txt_save_path', help='path to save the final txt triplet file', default='test_inner_new_' + SeqNum +'.txt')
parser.add_argument('--NumPositives' , help='number of positive samples for a given query', default=10)
parser.add_argument('--positive_thereshold', help='the threshold (in meters) below which we can consider a given samples as positive', default=4)
parser.add_argument('--max_negative_value', help='maximum value to be considered negative', default=6)
parser.add_argument('--min_negative_value', help='minimum value to be considered negative', default=4.1)
parser.add_argument('--DEM_img_path', help='the the path to DEMs for visulization', default='/home/sudarshan/CollaborativeSLAM_ws/src/CoSLAM/Experimental/CreateDEM/DEM/')
parser.add_argument('--reject_immediate_past', help='reject the immediate scans while considering positive samples', default=51)
args = parser.parse_args()


poses = np.loadtxt(args.poses_file)[ : , [3,7,11]]
f = open(args.txt_save_path, 'w')


def GetTriplets( QPose , pose_index ):

	Distance = np.linalg.norm( poses - QPose , axis =1)

	# print(Distance)

	indices = np.argwhere( Distance<args.positive_thereshold )

	Positives = []
	Negatives = []
	
	 

	for k in range(args.NumPositives ):
	
		if( k <  len( indices) ):

			if( indices[k] < pose_index - args.reject_immediate_past or indices[k] > pose_index + args.reject_immediate_past ):

				if(  np.linalg.norm( poses[indices[k]] - QPose) > 0.1 ):

					# print( "positive " + str(Distance[indices[k]]))
					Positives.append( indices[k] )

	Cntr = 0 

	for k in range( len(poses) ):

		if( Distance[k] < args.max_negative_value and Distance[k] > args.min_negative_value  ):

			# print( "negative" + str(Distance[k]))

			Negatives.append(k)
			if( Cntr == args.NumPositives -1 ):
				break 

			Cntr +=1 


	return Positives , Negatives


SeqNum = SeqNum +"/"
for i in tqdm(range( 1, len(poses ) , 1 )):

	QPose = poses[i]

	Positives , Negatives = GetTriplets(QPose , i)

	A = SeqNum + "000" +str(i) +".png" 
	anchor_path = os.path.join( args.DEM_img_path , A )

	for l in range(len(Positives)):
		B = SeqNum + "000" +str(Positives[l][0]) +".png"
		C =  SeqNum + "000" +str(Negatives[l]) +".png"

		triplet = str(A) + "," + str(B) +"," + str(C) + "\n"
		f.write(triplet)

		# positive_path = os.path.join( args.DEM_img_path , B )
		# negative_path = os.path.join( args.DEM_img_path , C )

		# fig = plt.figure() 
		# ax1 = fig.add_subplot(1,3,1)
		# ax2 = fig.add_subplot(1,3,2)
		# ax3 = fig.add_subplot(1,3,3)

		# AnchorImg = np.asarray( cv2.imread( anchor_path ,-1) )
		# PositiveImg = np.asarray(cv2.imread(positive_path , -1) )
		# NegativeImg = np.asarray(cv2.imread(negative_path , -1) )

		# print(anchor_path)
		# print( positive_path )
		# print( negative_path)


		# ax1.imshow( AnchorImg )
		# ax1.title.set_text("Anchor DEM ")

		# ax2.imshow( PositiveImg )
		# ax2.title.set_text("Positive DEM")

		# ax3.imshow( NegativeImg )
		# ax3.title.set_text("Negative DEM")

		# plt.show()

	# print(len(Positives) , len(Negatives))







