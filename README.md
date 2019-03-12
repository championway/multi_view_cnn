# Multi-view CNN 
## Brief Summary
* Last updated: 2019/01/30 with Pytorch v4.0

This repository is for point cloud classification from LIDAR. At first, we use PCL(Point Cloud Library) to do some point cloud preprocessing. After that, we project 3D point cloud to 2D image space and use CNN to do classification.

This project not only been tested in the simulator(Gazebo) but also use in 2018 [RobotX Challange](https://robotx.org/index.php), and it performed well during competition.

## Environment
- Ubuntu 16.04 LTS
- Point Cloud Library (PCL)
- ROS Kinetic
- Pytorch 4.0

[![](https://github.com/championway/multi_view_cnn/blob/master/pictures/youtube.png)](https://www.youtube.com/watch?v=-llRCISkNYE&t=1s)