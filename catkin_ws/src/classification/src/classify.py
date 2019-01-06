#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from robotx_msgs.msg import PCL_points, ObjectPose, ObjectPoseList
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from PIL import Image

class classify():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/obj_list', ObjectPoseList, self.call_back, queue_size = 1, buff_size = 2**24)
		self.pub_obj = rospy.Publisher("/obj_list/classify", ObjectPoseList, queue_size = 1)
		self.pub_marker = rospy.Publisher("/obj_classify", MarkerArray, queue_size = 1)
		#rospy.Subscriber('/pcl_array', PoseArray, self.call_back)
		self.boundary = 50
		self.height = 227.0
		self.width = 227.0
		self.point_size = 4	# must be integer
		self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		#self.index = 0

		# ***************************************************************
		# Get the position of caffemodel folder
		# ***************************************************************
		#self.model_name = rospy.get_param('~model_name')
		model_name = "robotx_ch3_epoch25"
		rospy.loginfo('[%s] model name = %s' %(self.node_name, model_name))
		rospack = rospkg.RosPack()
		model_path = rospack.get_path('classification') + '/model/' + model_name + '.pth'
		self.labels = ['buoy', 'dock', 'light_buoy', 'totem']

		# ***************************************************************
		# Variables
		# ***************************************************************
		self.bridge = CvBridge()
		self.pred_count = 0
		self.dim = (227, 227)
		self.img = None

		# ***************************************************************
		# Set up deeplearning model
		# ***************************************************************
		self.data_transform = transforms.Compose([ \
            transforms.Resize(227), \
            #transforms.RandomHorizontalFlip(), \
            transforms.ToTensor(), \
            transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                 std=[0.229, 0.224, 0.225])])
		self.model = torchvision.models.alexnet(pretrained = False)
		self.model.classifier[6] = nn.Linear(4096, 4)
		self.model = self.model.cuda()
		self.model.load_state_dict(torch.load(model_path))
		print(self.model)
		#self.model =  torch.load(model_path)
		

	def call_back(self, msg):
		obj_list = msg
		cluster_num = obj_list.size
		for i in range(cluster_num):
			tf_points = PoseArray()
			tf_points = obj_list.list[i].pcl_points
			centroids = Point()
			centroids = obj_list.list[i].position
			self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
			plane_xy = []
			plane_yz = []
			plane_xz = []
			pcl_size = len(tf_points.poses)

			# ======= Coordinate transform for better project performance ======
			position = [0, 0, 0]
			rad = math.atan2(centroids.y, centroids.x)
			quaternion = tf.transformations.quaternion_from_euler(0., 0., -rad)
			transformer = tf.TransformerROS()
			transpose_matrix = transformer.fromTranslationRotation(position, quaternion)
			for m in range(pcl_size):
				new_x = tf_points.poses[m].position.x
				new_y = tf_points.poses[m].position.y
				new_z = tf_points.poses[m].position.z
				orig_point = np.array([new_x, new_y, new_z, 1])
				new_center = np.dot(transpose_matrix, orig_point)
				tf_points.poses[m].position.x = new_center[0]
				tf_points.poses[m].position.y = new_center[1]
				tf_points.poses[m].position.z = new_center[2]

			# ======= project to XY, YZ, XZ plane =======
			for j in range(pcl_size):
				plane_xy.append([tf_points.poses[j].position.x, tf_points.poses[j].position.y])
				plane_yz.append([tf_points.poses[j].position.y, tf_points.poses[j].position.z])
				plane_xz.append([tf_points.poses[j].position.x, tf_points.poses[j].position.z])
			self.toIMG(pcl_size, plane_xy, 'xy')
			self.toIMG(pcl_size, plane_yz, 'yz')
			self.toIMG(pcl_size, plane_xz, 'xz')
			model_type = self.classify()

			# ***************************************************************
			# Add to object list
			# ***************************************************************
			obj_list.list[i].type = model_type
			#cv2.imwrite( "Image" + str(self.index) + ".jpg", self.image)
			#self.index = self.index + 1
			#print "Save image"
		self.pub_obj.publish(obj_list)
		self.drawRviz(obj_list)

	def toIMG(self, pcl_size, pcl_array, plane):
		min_m = 10e5
		min_n = 10e5
		max_m = -10e5
		max_n = -10e5
		for i in range(pcl_size):
			if min_m > pcl_array[i][0]:
				min_m = pcl_array[i][0]
			if min_n > pcl_array[i][1]:
				min_n = pcl_array[i][1]
			if max_m < pcl_array[i][0]:
				max_m = pcl_array[i][0]
			if max_n < pcl_array[i][1]:
				max_n = pcl_array[i][1]

		m_size = max_m - min_m
		n_size = max_n - min_n
		max_size = None
		min_size = None
		shift_m = False
		shift_n = False
		if m_size > n_size:
			max_size = m_size
			min_size = n_size
			shift_n = True
		else:
			max_size = n_size
			min_size = m_size
			shift_m = True
		scale = float((self.height-self.boundary*2)/max_size)
		shift_size = int(round((self.height - self.boundary*2 - min_size*scale)/2))
		img = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		for i in range(pcl_size):
			if shift_m:
				pcl_array[i][0] = int(round((pcl_array[i][0] - min_m)*scale)) + shift_size + self.boundary
				pcl_array[i][1] = int(round((pcl_array[i][1] - min_n)*scale)) + self.boundary
			elif shift_n:
				pcl_array[i][0] = int(round((pcl_array[i][0] - min_m)*scale)) + self.boundary
				pcl_array[i][1] = int(round((pcl_array[i][1] - min_n)*scale)) + shift_size + self.boundary
			for m in range(-self.point_size, self.point_size + 1):
				for n in range(-self.point_size, self.point_size + 1):
					img[pcl_array[i][0] + m  , pcl_array[i][1] + n] = (0,255,255)
					if plane == 'xz':
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][0] = 255
					elif plane == 'yz':
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][1] = 255
					elif plane == 'xy':
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][2] = 255
		#cv2.imwrite( "Image_" + plane + ".jpg", img )

	def classify(self):
		# ***************************************************************
		# Using Pytorch Model to do prediction
		# ***************************************************************
		cv_img = cv2.resize(self.image, self.dim)
		pil_img = Image.fromarray(cv_img.astype('uint8'))
		torch_img = self.data_transform(pil_img)
		torch_img = np.expand_dims(torch_img, axis=0)
		input_data = torch.tensor(torch_img).type('torch.FloatTensor').cuda()
		t_start = time.clock()
		output = self.model(input_data)
		pred_y = int(torch.max(output, 1)[1].cpu().data.numpy())
		#print "prediction time taken = ", time.clock() - t_start
		#print "Predict: ", self.labels[output_max_class]
		#print output_prob[output_max_class]
		#if output_prob[output_max_class]<0.7:
		#	return "None"
		return self.labels[pred_y]

	def drawRviz(self, obj_list):
		marker_array = MarkerArray()
		# marker_array.markers.resize(obj_list.size)
		for i in range(obj_list.size):
			marker = Marker()
			marker.header.frame_id = obj_list.header.frame_id
			marker.id = i
			marker.header.stamp = rospy.Time.now()
			marker.type = Marker.CUBE
			marker.action = Marker.ADD
			marker.lifetime = rospy.Duration(0.5)
			marker.pose.position.x = obj_list.list[i].position.x
			marker.pose.position.y = obj_list.list[i].position.y
			marker.pose.position.z = obj_list.list[i].position.z
			marker.pose.orientation.x = 0.0
			marker.pose.orientation.y = 0.0
			marker.pose.orientation.z = 0.0
			marker.pose.orientation.w = 1.0
			marker.scale.x = 1
			marker.scale.y = 1
			marker.scale.z = 1
			if obj_list.list[i].type == "buoy":
				marker.color.r = 0
				marker.color.g = 0
				marker.color.b = 1
				marker.color.a = 0.5
			elif obj_list.list[i].type == "totem":
				marker.color.r = 0
				marker.color.g = 1
				marker.color.b = 0
				marker.color.a = 0.5
			elif obj_list.list[i].type == "light_buoy":
				marker.color.r = 1
				marker.color.g = 1
				marker.color.b = 0
				marker.color.a = 0.5
			elif obj_list.list[i].type == "dock":
				marker.color.r = 1
				marker.color.g = 1
				marker.color.b = 1
				marker.color.a = 0.5
				marker.scale.x = 6
				marker.scale.y = 6
				marker.scale.z = 1
			elif obj_list.list[i].type == "deliver":
				marker.color.r = 0
				marker.color.g = 1
				marker.color.b = 1
				marker.color.a = 0.5
				marker.scale.x = 6
				marker.scale.y = 6
				marker.scale.z = 1
			else:
				marker.color.r = 0
				marker.color.g = 0
				marker.color.b = 0
				marker.color.a = 0.5
			marker_array.markers.append(marker)
		self.pub_marker.publish(marker_array)

if __name__ == '__main__':
	rospy.init_node('classify')
	foo = classify()
	rospy.spin()