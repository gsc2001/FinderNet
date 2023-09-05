# FinderNet: A Data Augmentation Free Canonicalization aided Loop Detection and Closure technique for Point clouds in 6-DOF separation.

## Abstract

We focus on the problem of LiDAR point cloud based loop detection (or Finding) and closure (LDC) for mobile robots. State-of-the-art (SOTA) methods directly generate learned embeddings from a given point cloud, require large data transfers, and are not robust to wide viewpoint variations in 6 Degrees-of-Freedom (DOF). Moreover, the absence of strong priors in an unstructured point cloud leads to highly inaccurate LDC. In this original approach, we propose independent roll and pitch canonicalization of point clouds using a common dominant ground plane. We discretize the canonicalized point clouds along the axis perpendicular to the ground plane leads to images similar to digital elevation maps (DEMs), which expose strong spatial priors in the scene. Our experiments show that LDC based on learnt embeddings from such DEMs is not only data efficient but also significantly more robust, and generalizable than the current SOTA. We report an (average precision for loop detection, mean absolute translation/rotation error) improvement of (11.0, 34.0/25.4)% on GPR10 sequence, over the current SOTA. To further test the robustness of our technique on point clouds in 6-DOF motion we create and opensource a custom dataset called Lidar-UrbanFly Dataset (LUF) which consists of point clouds obtained from a LiDAR mounted on a quadrotor

## In action

<iframe width="560" height="315" src="https://www.youtube.com/embed/1P6JMqbb_sM?si=DDAoskWwFUXruj8H" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Supplementary material: [link](https://drive.google.com/file/d/1LibsAML9UY7IgZPuWi8RPVeyYQW-VWfS/view?usp=sharing)

PrePrint: https://arxiv.org/pdf/2304.01074.pdf
