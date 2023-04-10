import numpy as np 
import open3d as o3d
import copy
import math
import matplotlib.pyplot as plt 
import cv2
import argparse
from scipy.optimize import minimize
# import torch
# import Canonicalize as Can
import utilities as ut
import DEMCreator as DC


parser = argparse.ArgumentParser(description='DEM creation for training')
parser.add_argument('--PCD_path', help='path to the file of point clouds' , default='/home/sudarshan/test_es/src/KITTI/avg_kitti/dataset/sequences/08/velodyne')
parser.add_argument('--DEM_save_path', help='path to save the original DEM', default='/home/sudarshan/CollaborativeSLAM_ws/src/CoSLAM/Experimental/CreateDEM/DEM/08_augument')
parser.add_argument('--augument_angle_save_path', help='path to save augumented angles' , default='/home/sudarshan/CollaborativeSLAM_ws/src/CoSLAM/Experimental/augument.txt')
args = parser.parse_args()

PCDPathList = ut.GetPointClouds( args.PCD_path) 

# print("Creating Point Cloud and Pose mapping")

# PointCloud_pose_dict = ut.GetPCD_pose_map(  PCDPathList , args.pose_path, args.NumSamples)

# print(" Creating the Dataset ")

# Can.CanonicalizeDEM(PointCloud_pose_dict, PCDPathList , args.original_save_path , 
# 	args.augumented_save_path , args.canonicalized_save_path , args.train_csv_file)



print("Creating DEM for sequence 08 " )


# print( PCDPathList)

DC.CreateDEM(PCDPathList , args.DEM_save_path, args.augument_angle_save_path)