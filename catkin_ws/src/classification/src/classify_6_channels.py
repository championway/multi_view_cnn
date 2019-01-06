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
from os.path import expanduser
caffe_root = expanduser("~")
sys.path.insert(0, caffe_root + '/caffe/python')
import caffe
import os
from PIL import Image

class classify_6_channels():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/obj_list/roi', ObjectPoseList, self.call_back)
		self.pub_obj = rospy.Publisher("/obj_list/classify", ObjectPoseList, queue_size = 1)
		self.pub_marker = rospy.Publisher("/obj_classify", MarkerArray, queue_size = 1)
		#rospy.Subscriber('/pcl_array', PoseArray, self.call_back)
		self.boundary = 50
		self.height = 227.
		self.width = 227.
		self.point_size = 4	# must be integer
		self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		self.no_camera_img = False
		#self.index = 0

		# ***************************************************************
		# Get the position of caffemodel folder
		# ***************************************************************
		#self.model_name = rospy.get_param('~model_name')
		self.model_name = "caffenet_6_channels"
		self.prototxt_name = "caffenet_6_channels"
		self.model_path = "/object_classification/caffenet_6_channels/"
		rospy.loginfo('[%s] model name = %s' %(self.node_name, self.model_name))
		rospack = rospkg.RosPack()
		self.model_Base_Dir = rospack.get_path('dl_models') + self.model_path
		self.labels = []
		with open(self.model_Base_Dir+'lab_list_6_channels.txt', 'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.replace('\n', '')
			self.labels.append(line)

        # ***************************************************************
        # Variables
        # ***************************************************************
		self.bridge = CvBridge()
		self.pred_count = 0
		self.dim = (227, 227)
		self.img = None
		self.roi = None

		# ***************************************************************
		# Set up caffe
		# ***************************************************************

		self.model_def = self.model_Base_Dir + self.prototxt_name + '.prototxt'
		self.model_weights = self.model_Base_Dir + self.model_name + '.caffemodel'
		self.net = caffe.Net(	self.model_def,        # defines the structure of the model
								self.model_weights,    # contains the trained weights
								caffe.TEST)

	def call_back(self, msg):
		obj_list = ObjectPoseList()
		obj_list = msg
		cluster_num = obj_list.size

		#tf_points = PCL_points()
		#tf_points = msg
		#cluster_num = len(tf_points.list)
		#pcl_size = len(msg.poses)
		for i in range(cluster_num):
			self.no_camera_img = False
			self.get_roi_image(obj_list.list[i].img)
			self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
			tf_points = PoseArray()
			tf_points = obj_list.list[i].pcl_points
			centroids = Point()
			centroids = obj_list.list[i].position
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
			# Add typpe to object list
			# ***************************************************************
			obj_list.list[i].type = model_type
			#cv2.imwrite( "Image" + str(self.index) + ".jpg", self.image)
			#self.index = self.index + 1
			#print "Save image"
		self.pub_obj.publish(obj_list)
		self.drawRviz(obj_list)

	def create_black_img(self):
		self.no_camera_img = True
		print "Skip"
		black_img = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		return black_img

	def get_roi_image(self, ros_img):
		if ros_img != None and ros_img.height != 0 and ros_img.width!=0:
			try:
				cv_image = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
				self.roi = self.resize_keep_ratio(cv_image, int(self.width), int(self.height))
				#cv2.imwrite( "roi/Image" + str(self.index) + ".jpg", img)
			except CvBridgeError as e:
				print (e)
				print ("CvBridge Error!!!")
				self.roi = self.create_black_img()
		else:
			print "No camera image, fill with black image!!!"
			self.roi = self.create_black_img()

	def resize_keep_ratio(self, img, width, height, fill_color=(0, 0, 0, 0)):
		#======= Make sure image is smaller than background =======
		h, w, channel = img.shape
		h_ratio = float(float(height)/float(h))
		w_ratio = float(float(width)/float(w))
		#if h_ratio <= 1 or w_ratio <= 1:
		ratio = h_ratio
		if h_ratio > w_ratio:
			ratio = w_ratio
		img = cv2.resize(img,(int(ratio*w),int(ratio*h)))
		#======== Paste image to background =======
		im = Image.fromarray(np.uint8(img))
		x, y = im.size
		new_im = Image.new('RGBA', (width, height), fill_color)
		new_im.paste(im, ((width - x) / 2, (height - y) / 2))
		img_4 = np.asarray(new_im)
		#image_ = np.array(image)
		img_3 = img_4[...,:3]
		return img_3

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

	def concatenation_img(self, img1, img2):
		img = np.dstack((img1, img2))
		#img = img[:,:,::-1]
		mean = (104.00699, 116.66877, 122.67892, 104.00699, 116.66877, 122.67892)
		#img = img - np.array(mean)
		#img = img.transpose((2,0,1))
		return img

	def classify(self):
		# ***************************************************************
		# Using Caffe Model to do prediction
		# ***************************************************************
		img1 = cv2.resize(self.image, self.dim)
		img2 = cv2.resize(self.roi, self.dim)
		#img1 = np.zeros((227, 227, 3), np.uint8)
		#img2 = np.zeros((227, 227, 3), np.uint8)
		img = self.concatenation_img(img1, img2)
		#top[0].reshape(1, *img.shape)
		#top[0].data[...] = img

		caffe.set_device(0)
		caffe.set_mode_gpu()
		t_start = time.clock()
		transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2, 0, 1))
		self.net.blobs['data'].reshape(1, 6, self.dim[0], self.dim[1])
		self.net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = self.net.forward()
		output_prob = output['prob'][0]
		output_max_class = output_prob.argmax()
		#print output_prob
		#print "prediction time taken = ", time.clock() - t_start
		#print "Predict: ", self.labels[output_max_class]
		if output_prob[output_max_class]<0.9:
			return "None"
		return self.labels[output_max_class]

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
			elif obj_list.list[i].type == "totem_red":
				marker.color.r = 1
				marker.color.g = 0
				marker.color.b = 0
				marker.color.a = 0.5
			elif obj_list.list[i].type == "totem_green":
				marker.color.r = 0
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
			else:
				marker.color.r = 0
				marker.color.g = 0
				marker.color.b = 0
				marker.color.a = 0.5
			marker_array.markers.append(marker)
		self.pub_marker.publish(marker_array)

if __name__ == '__main__':
	rospy.init_node('classify_6_channels')
	foo = classify_6_channels()
	rospy.spin()