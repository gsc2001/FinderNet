import numpy as np
import pandas as pd
import os
import glob 
import open3d as o3d 
from dgl.geometry import farthest_point_sampler
import torch
import cv2
from PIL import Image
import PIL
import random
import math
import torch.nn as nn
import matplotlib.pyplot as plt



def GetData( AnchorPCDPath, PositivePCDPath, NegativePCDPath , BaseDataPath , GridSize ,  RefPCD , NumPts_in_Grid , BatchSize ):

	numPCDs = len( AnchorPCDPath) 

	AnchorImgs = []
	PositiveImgs = []
	NegativeImgs = [] 

	for (a,p,n) in zip(  AnchorPCDPath , PositivePCDPath ,  NegativePCDPath ): 


		anchor_PCD_path = BaseDataPath + str( a )
		positive_PCD_path = BaseDataPath + str(p )
		negative_PCD_path = BaseDataPath + str(n)


		AnchorPCD_i =  o3d.io.read_point_cloud(anchor_PCD_path)
		PositivePCD_i = o3d.io.read_point_cloud(positive_PCD_path)
		NegativePCD_i =  o3d.io.read_point_cloud(negative_PCD_path)




		CanocalizedAnchor , CanonicalizedPositive , CanonicalizedNegative , PlaneModel = Can.GetCanonicalizedPCD( AnchorPCD_i , PositivePCD_i , NegativePCD_i , RefPCD  )
		# o3d.visualization.draw_geometries([ CanocalizedAnchor]) 
		# print(" result of canonicalization")
		AnchorImg ,PositiveImg , NegativeImg = Pimg.ProjectPoints( CanocalizedAnchor ,  CanonicalizedPositive, CanonicalizedNegative , GridSize , NumPts_in_Grid  ,  PlaneModel)

		AnchorImgs.append( AnchorImg)
		PositiveImgs.append( PositiveImg )
		NegativeImgs.append( NegativeImg )


	AnchorImgs =  np.asarray(AnchorImgs) #torch.tensor( np.asarray(AnchorImgs) , dtype = torch.float  ).to("cuda")
	PositiveImgs = np.asarray(PositiveImgs) #torch.tensor( np.asarray(PositiveImgs) , dtype = torch.float ).to("cuda")
	NegativeImgs =np.asarray(NegativeImgs)  #torch.tensor( np.asarray(NegativeImgs) , dtype= torch.float ).to("cuda")


	return AnchorImgs , PositiveImgs , NegativeImgs


def Readimages( ImageIndices , AnchorRootPath , PositiveRootPath  ):

	NumImages = len(ImageIndices)
	AnchorImgs = []
	PositiveImgs = []
	NegativeImgs = [] 

	GT_Yaw_values = torch.zeros( ( NumImages , 2 ) )

	for i in range(NumImages):

		anchor_img_name = str(ImageIndices[i]) + ".png"
		positive_img_name =  str(ImageIndices[i]) + ".png"
		negative_img_name = str(ImageIndices[i]) + ".png"

		AnchorPath = os.path.join(AnchorRootPath , anchor_img_name)
		PositivePath = os.path.join(PositiveRootPath , positive_img_name)

		anchor_img =  np.asarray( Image.open( AnchorPath )).astype(np.float32)
		positive_img =  np.asarray( Image.open( PositivePath )).astype(np.float32)
		col_num   = 0 #random.randint( 0, 10 ) 
		center = (501/2, 501/2)


		rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=col_num, scale=1)

		# print(col_num)
		# print(anchor_img)
		# print("-------")
		positive_img = cv2.warpAffine(src=anchor_img, M=rotate_matrix, dsize=(501, 501))

		GT_Yaw_values[i][0] = math.cos(col_num*3.14/180)
		GT_Yaw_values[i][1] = math.sin(col_num*3.14/180)

		AnchorMaxRadius = np.sqrt(((anchor_img.shape[0]/2.0)**2.0)+((anchor_img.shape[1]/2.0)**2.0))
		PositiveMaxRadius = np.sqrt( ( (positive_img.shape[0]/2.0)**2.0)+((positive_img.shape[1]/2.0)**2.0))

		AnchorPolarImg = cv2.linearPolar(anchor_img,(anchor_img.shape[0]/2, anchor_img.shape[1]/2), AnchorMaxRadius, cv2.WARP_FILL_OUTLIERS)
		PositivePolarImg = cv2.linearPolar(positive_img,(positive_img.shape[0]/2, positive_img.shape[1]/2), PositiveMaxRadius, cv2.WARP_FILL_OUTLIERS)

		AnchorPolarImg = AnchorPolarImg.astype(np.uint8).T
		PositivePolarImg = PositivePolarImg.astype(np.uint8).T

		AnchorImgs.append(AnchorPolarImg)
		PositiveImgs.append(PositivePolarImg)

	PositiveImgs = torch.tensor( np.asarray(PositiveImgs) , dtype = torch.float  ).to("cuda") #np.asarray(PositiveImgs)
	AnchorImgs = torch.tensor( np.asarray(AnchorImgs) , dtype = torch.float  ).to("cuda") #np.asarray(AnchorImgs)

	return AnchorImgs , PositiveImgs  , (GT_Yaw_values).to("cuda")



def CircularShiftIMG( input_img, shift_num ):

	NumCols = np.shape(input_img)[1]
	Numrows = np.shape(input_img)[0]
	new_img = np.zeros((Numrows,NumCols))
	
	for i in range(NumCols):

		new_col = int( ( i + shift_num)%NumCols)
		#print( new_img[:, new_col ] )

		new_img[:, new_col ] = input_img[:, i ]

	return new_img


def ReadImages2( ImagePaths , isNegative ):

	NumImages = len(ImagePaths)

	GT_Yaw_values = torch.zeros( ( NumImages , 360 ) )
	AnchorImgs = []
	PositiveImgs = []

	for i in range(NumImages):

		H = 100
		W = 400


		IMG = Image.open( ImagePaths[i] )
		W, H = IMG.size
		IMG =  np.asarray( IMG ).astype(np.float32) 
		IMG = cv2.resize(IMG, (500, 500), interpolation = cv2.INTER_NEAREST)



		AnchorPolarImg = IMG.astype(np.uint8)
		AnchorImgs.append(AnchorPolarImg)

	if( isNegative ):

		AnchorImgs = torch.tensor( np.asarray(AnchorImgs) ,  dtype = torch.float).to("cuda")

	return AnchorImgs

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def ReadImages3( ImagePaths , isNegative , Rotate , value ):

	NumImages = len(ImagePaths)

	GT_Yaw_values = torch.zeros( ( NumImages , 360 ) )
	AnchorImgs = []
	PositiveImgs = []

	for i in range(NumImages):

		H = 100
		W = 400


		IMG = Image.open( ImagePaths[i] )
		W, H = IMG.size
		IMG =  np.asarray( IMG ).astype(np.float32) 
		IMG = cv2.resize(IMG, (500, 500), interpolation = cv2.INTER_NEAREST)

		if(Rotate):

			r1 = value #random.randint(0, 359)
			IMG = rotate_image( IMG, r1)

		AnchorPolarImg = IMG.astype(np.uint8)
		AnchorImgs.append(AnchorPolarImg)

	if( isNegative ):

		AnchorImgs = torch.tensor( np.asarray(AnchorImgs) ,  dtype = torch.float).to("cuda")

	return AnchorImgs



def GetSIFTCorrespondance( AnchorImgs, PositiveImgs , NegativeImgs):

	NumImages , W, H = np.shape( AnchorImgs ) 

	AnchorIMGS2 = []
	PositiveIMGS2 = []
	NegativeIMGS2 = []

	Image_a_feature_points =[]
	Image_p_feature_points =[]
	M = nn.Sigmoid()
	sift = cv2.SIFT_create()
	bf = cv2.BFMatcher()

	KeyPoints_a = np.zeros( ( NumImages , 100 , 3 ) )
	KeyPoints_p = np.zeros( ( NumImages , 100 , 3 ) ) 


	for i in range(NumImages):

		Img1 = M( torch.tensor(AnchorImgs[i])).numpy()
		Img2= M( torch.tensor(PositiveImgs[i])).numpy()

		Img1 = cv2.normalize(Img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
		Img2 = cv2.normalize(Img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


		keypoints_a, descriptors_a = sift.detectAndCompute( Img1 , None)
		keypoints_p, descriptors_p = sift.detectAndCompute(Img2 , None)
		matches = bf.knnMatch(descriptors_a,descriptors_p, k=2)
	
		good = []

		pts_a = []
		pts_p = []

		W1, H1 = np.shape(Img1)
		W2, H2 = np.shape(Img2)
		cntr =0 

		for m,n in matches:

			if m.distance < 0.6*n.distance:
				good.append([m])
				img1_idx = m.queryIdx
				img2_idx = m.trainIdx
				(x1, y1) = keypoints_a[img1_idx].pt
				(x2, y2) = keypoints_p[img2_idx].pt

				A1 = [x1 - W1/2, y1 -H1/2  , 1]
				A2 =  [x2 - W1/2 , y2 -H1/2 , 1] 

				KeyPoints_a[i][cntr] = A1 / np.linalg.norm( [ W1/2, H1/2] )
				KeyPoints_p[i][cntr] = A2/ np.linalg.norm( [ W2/2, H2/2] )

				cntr += 1

				if( cntr >99):
					break

				# pts_a.append( A1)
				# pts_p.append( A2 )

		if( len(good) > 3):

			AnchorIMGS2.append(AnchorImgs[i])
			PositiveIMGS2.append(PositiveImgs[i])
			NegativeIMGS2.append( NegativeImgs[i])


	KeyPoints_a = torch.tensor(KeyPoints_a).to("cuda")
	KeyPoints_p = torch.tensor(KeyPoints_p).to("cuda")

	AnchorImgs = torch.tensor( np.asarray(AnchorIMGS2) ,  dtype = torch.float).to("cuda")
	PositiveImgs = torch.tensor( np.asarray(PositiveIMGS2) ,  dtype = torch.float).to("cuda")
	NegativeImgs = torch.tensor( np.asarray(NegativeIMGS2) ,  dtype = torch.float).to("cuda")

	return AnchorImgs, PositiveImgs, NegativeImgs , KeyPoints_a , KeyPoints_p




def ReadTopViewImgs( AnchorImgPaths , PositiveImgPaths, NegativeImgPaths ):

	NumImages = len(AnchorImgPaths)

	AnchorImgs1 = []
	PositiveImgs1 = []
	NegativeImgs1 = []

	for i in range(NumImages):

		AIMG =  np.asarray(  Image.open( AnchorImgPaths[i] ) ).astype(np.float32)#.astype(np.uint8).T
		PIMG = np.asarray(    Image.open( PositiveImgPaths[i]  ) ).astype(np.float32)#.astype(np.uint8).T
		NIMG = np.asarray( Image.open( NegativeImgPaths[i]  )).astype(np.float32)#.astype(np.uint8).T

		AIMG = cv2.resize(AIMG, (303, 303), interpolation = cv2.INTER_NEAREST)
		PIMG = cv2.resize(PIMG, (303, 303), interpolation = cv2.INTER_NEAREST)
		NIMG = cv2.resize(NIMG, (303, 303), interpolation = cv2.INTER_NEAREST)

		AnchorImgs1.append(AIMG)
		PositiveImgs1.append(PIMG)
		NegativeImgs1.append(NIMG)


	AnchorImgs1 = torch.tensor( np.asarray(AnchorImgs1) ,  dtype = torch.float).to("cuda")
	PositiveImgs1 = torch.tensor( np.asarray(PositiveImgs1) ,  dtype = torch.float).to("cuda")
	NegativeImgs1 = torch.tensor( np.asarray(NegativeImgs1) ,  dtype = torch.float).to("cuda")

	return AnchorImgs1 ,   PositiveImgs1  , NegativeImgs1



def ReadTopViewImgsInference( DataPaths, start_index , batch_size ):

	Data = []

	for i in range( int(start_index) ,int( start_index + batch_size) ):

		IMG =  np.asarray(  Image.open( DataPaths[i] ) ).astype(np.float32)
		IMG = cv2.resize(IMG, (303, 303), interpolation = cv2.INTER_NEAREST)
		Data.append(IMG)


	Data = torch.tensor( np.asarray(Data) ,  dtype = torch.float).to("cuda")

	return Data











