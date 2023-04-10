import numpy as np
import os
import math
import copy
import open3d as o3d
import matplotlib.pyplot as plt 
import random
import imageio
import cv2 
from tqdm import tqdm


def FilterPCD( PCD ):

	Pts = PCD.points
	mean = np.mean(Pts , axis = 0)
	Pts = Pts -mean
	PCD.points = o3d.utility.Vector3dVector( Pts)

	# pcd_tree = o3d.geometry.KDTreeFlann(PCD)
	# [k, idx, _] = pcd_tree.search_radius_vector_3d([0,0,0], 25.0)

	NewPCD = o3d.geometry.PointCloud()

	# NewPCD.points = o3d.utility.Vector3dVector( np.asarray(PCD.points)[idx[1:], :] )

	# mean = np.mean(np.asarray(NewPCD.points ) , axis = 0)
	# Pts = np.asarray(NewPCD.points ) - mean 
	# NewPCD.points = o3d.utility.Vector3dVector( Pts)


	# Pts = np.asarray( PCD.points )
	Distance_from_Center = np.linalg.norm(Pts , axis=1)

	indices = np.argwhere( Distance_from_Center < 25 )

	NewPCD = o3d.geometry.PointCloud()
	NumPts = len(indices )
	Points  = np.reshape(  Pts[indices] , ( NumPts , 3) )
	NewPCD.points = o3d.utility.Vector3dVector( Points )


	return NewPCD

def InitilizeYawPitch( Normal ):

	beta = np.arctan( - Normal[0]/Normal[2])

	K1 = np.sin(beta)
	K2 = np.cos(beta)

	gamma = np.arctan( Normal[1]/ (  Normal[2]*K2 - Normal[0]*K1 ) )

	C= np.cos 
	S = np.sin 

	R = np.asarray([ [ C(beta) , 0 , S(beta) ] , 
					[ S(gamma)*S(beta) , C(gamma) , -S(gamma)*C(beta) ] , 
					[ -S(beta)*C(gamma) , S(gamma) , C(gamma)*C(beta) ] ])

	return R 



def GetDEM( PCD ):
	
	OriginalPCD = copy.deepcopy(PCD)
	PlanePointCLoud =  copy.deepcopy(PCD)
	plane_model, inliers = PCD.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)

	R_init = InitilizeYawPitch(plane_model  )

	N = plane_model[0:3]

	PlanePointCLoud = PCD.select_by_index(inliers)

	PlanePointCLoud2 = copy.deepcopy(PlanePointCLoud)

	PlanePoints = np.asarray(PlanePointCLoud.points)
	Center_plane =  np.mean( np.asarray( PlanePoints) , axis =0 )
	PlanePoints -= np.mean( PlanePoints , axis =0 )

	NumPlanePts = len(PlanePoints)

	for i in range( NumPlanePts ):


		PlanePoints[i] = PlanePoints[i]/ np.linalg.norm( PlanePoints[i] )

	PlanePointCLoud.points = o3d.utility.Vector3dVector(PlanePoints)
	


	H = np.eye(4)
	t = [0, 0, 0]
	H[0:3,0:3] = np.ones(3)
	H[0:3, 3] = t
	H[3, :] = [0, 0 ,0 ,1]

	threshold = 0.02

	trans_init = H


	Center = np.mean(  np.asarray(PlanePointCLoud2.points) , axis=0  )
	Z = 0

	Center= [0,0,0]


	OriginalPCD =  OriginalPCD.rotate(R_init, center=(Center[0], Center[1], Center[2] ))
	Pts = np.asarray(OriginalPCD.points)

	plane_model, inliers = OriginalPCD.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)


	xmax , xmin = np.amax( Pts[:, 0 ]  ) , np.amin( Pts[:, 0 ]  )
	ymax , ymin = np.amax( Pts[:, 1 ]  ) , np.amin( Pts[:, 1 ]  )
	zmax , zmin = np.amax( Pts[:, 2 ]  ) , np.amin( Pts[:, 2 ]  )


	MaxNumGrid = 500
	resolution = 0.1 #( xmax - xmin)/NumGrids
	NumGrids = int( ( xmax - xmin)/resolution )

	HeightImage = np.zeros( ( MaxNumGrid , MaxNumGrid ) )

	A = np.asarray(plane_model)


	for pt in range(len(Pts)):


		x = min( int( (Pts[pt][0]  )/ resolution ) , NumGrids/2)    +250
		y = min(int( (Pts[pt][1] )/ resolution ) , NumGrids/2  ) +250
		Pt = [ Pts[pt][0] , Pts[pt][1] , Pts[pt][2] , 1 ]


		dist_from_plane = max( A.T@Pt , 0) *10

		if( (x < NumGrids) and (y < NumGrids) ):


			HeightImage[x][y] = max( dist_from_plane , HeightImage[x][y] )


	# HeightImage[225:250, 225:250] = 10


	R = np.eye(3)
	# print("-------------------------------")

	return R , Z , HeightImage, xmin , ymin , resolution


def GetDEMUnreal( PCD ):

	plane_model, inliers = PCD.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)

	R_init = InitilizeYawPitch(plane_model  )

	# print(plane_model )


	PCD =  PCD.rotate(R_init, center=(0, 0, 0 ))

	Pts = np.asarray(PCD.points)

	xmax , xmin = np.amax( Pts[:, 0 ]  ) , np.amin( Pts[:, 0 ]  )
	ymax , ymin = np.amax( Pts[:, 1 ]  ) , np.amin( Pts[:, 1 ]  )
	zmax , zmin = np.amax( Pts[:, 2 ]  ) , np.amin( Pts[:, 2 ]  )

	MaxNumGrid = 500
	resolution = 0.1 #( xmax - xmin)/NumGrids
	NumGrids = int( ( xmax - xmin)/resolution )

	HeightImage = np.zeros( ( MaxNumGrid , MaxNumGrid ) )
	plane_model, inliers = PCD.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)
	A = np.asarray(plane_model)
	# A[0] = 0 
	# A[1] =0
	# A[2] = 1 

	for pt in range(len(Pts)):


		x = min( int( (Pts[pt][0]  )/ resolution ) , NumGrids/2)    +250
		y = min(int( (Pts[pt][1] )/ resolution ) , NumGrids/2  ) +250
		Pt = [ Pts[pt][0] , Pts[pt][1] , Pts[pt][2] , 1 ]


		dist_from_plane = max( A.T@Pt , 0) *10

		# print(dist_from_plane)

		if( (x < NumGrids) and (y < NumGrids) ):


			HeightImage[x][y] = max( dist_from_plane , HeightImage[x][y] )	

	return HeightImage




def CreateDEM( PCDPathList,  save_path_original , TxtPath ):

	f = open(TxtPath, "w")
	for i in tqdm(range( 0 , len(PCDPathList) ) ) :

		bin_pcd = np.fromfile(PCDPathList[i] , dtype=np.float32)
		points = bin_pcd.reshape((-1, 4))[:, 0:3]
		o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

		roll = random.randint( -30,30 )
		pitch = random.randint( -30,30 )
		yaw_aug = random.randint( -30,30 )

		roll = roll*np.pi/180 
		pitch = pitch*np.pi/180 

		angle_write = str(roll) +" " +str(pitch) +"\n"

		f.write(angle_write)

		CenterPts = np.mean( np.asarray(o3d.utility.Vector3dVector(points)) , axis =1 )

		center = ( CenterPts[0] , CenterPts[1] ,CenterPts[2])


		R1_rp = o3d_pcd.get_rotation_matrix_from_yxz( ( roll, pitch , 0) )
		o3d_pcd = o3d_pcd.rotate(R1_rp, center=center)

		PCD = o3d_pcd

		NewPCD = FilterPCD(PCD)

		OriginalPCD = copy.deepcopy(NewPCD)


		R , Z , original_DEM, xmin , ymin , resolution = GetDEM(NewPCD)
		original_name = "000" +str(i)+".png"
		original_save_path = os.path.join(save_path_original , original_name)
		# cv2.imwrite(original_save_path, original_DEM)

		plt.imshow(original_DEM)
		plt.show()



def CreateDEMUnreal( PCDPathList  , save_path_original ):

	for i in tqdm(range( 0 , len(PCDPathList) ) ) :

		# print(PCDPathList[i])
		# print("---------------------")

		# bin_pcd = np.fromfile(PCDPathList[i] , dtype=np.float32)
		# print(bin_pcd[0])

		# print( np.shape( bin_pcd ))
		# points = bin_pcd.reshape((-1, 4))[:, 0:3]

		# print(points)

		points = np.load( PCDPathList[i] )

		R_frame_transform = np.asarray([[ 0,0,1 ] , [1,0,0] , [0,-1,0] ])

		points = (R_frame_transform@(points.T)).T


		o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

		o3d.visualization.draw_geometries([o3d_pcd])

		# CenterPts = np.mean( np.asarray(o3d.utility.Vector3dVector(points)) , axis =1 )
		# center = ( CenterPts[0] , CenterPts[1] ,CenterPts[2])

		NewPCD = FilterPCD(o3d_pcd)

		# o3d.visualization.draw_geometries([NewPCD])

		original_DEM   = GetDEMUnreal(NewPCD)

		original_name = "000" +str(i)+".png"

		original_save_path = os.path.join(save_path_original , original_name)

		cv2.imwrite(original_save_path, original_DEM)

		# plt.imshow(original_DEM)
		# plt.show()


		# o3d.visualization.draw_geometries([o3d_pcd])

		# print(points)
