# FinderNet: A Data Augmentation Free Canonicalization aided Loop Detection and Closure technique for Point clouds in 6-DOF separation.


### The page is under construction, please stay tuned 

We focus on the problem of LiDAR point cloud based loop detection (or Finding) and closure (LDC) in a multi-agent setting. State-of-the-art (SOTA) techniques directly generate learned embeddings of a given point cloud, require large data transfers, and are not robust to wide variations in 6 Degrees-of-Freedom (DOF) viewpoint. Moreover, absence of strong priors in an unstructured point cloud leads to highly inaccurate LDC. In this original approach, we propose independent roll and pitch canonicalization of the point clouds using a common dominant ground plane. Discretization of the canonicalized point cloud along the axis perpendicular to the ground plane leads to an image similar to Digital Elevation Maps (DEMs), which exposes strong spatial priors in the scene. Our experiments show that LDC based on learnt embeddings of such DEMs is not only data efficient but also significantly more robust, and generalizable than the current SOTA. We report significant performance gain in terms of Average Precision for loop detection and absolute translation/rotation error for relative pose estimation (or loop closure) on Kitti, GPR and Oxford Robot Car over multiple SOTA LDC methods. Our encoder technique allows to compress the original point cloud by over 830 times. To further test the robustness of our technique we create and opensource a custom dataset called Lidar-UrbanFly Dataset (LUF) which consists of point clouds obtained from a LiDAR mounted on a quadrotor.

![](https://github.com/gsc2001/FinderNet/blob/main/Images/Main.png)

### Installation Instructions

1. Install [PyTorch](https://pytorch.org/)
2. Install the requirements ```pip install -r requirements.txt```


### Canonicalization and DEM Creation 


### Download the Pre-Processed DEM

Please download the preprocessed DEM for various datasets in this [link](https://drive.google.com/drive/folders/19FZUBr8iLdD033HEz-rgrpoSSaydRb0F?usp=sharing)



### Training FinderNet
```
python3 Train/Train.py <Set the paths in the arguemnts>

```

### Inference 

```
python3 Inference/Inference2.py <Set the paths arguments>

```

Please download the pretrained models from the [link]()

### DEM Generation for Custom Dataset 


```
python3 DEM\ Generation/CreateDEM.py --PCD_path <Set the path to the directory of point clouds> --DEM_save_path <Path to save the resulting DEM>

```
