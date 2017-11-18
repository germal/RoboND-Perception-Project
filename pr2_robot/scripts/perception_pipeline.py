#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import os
import sys


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    #  Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    #  Statistical Outlier Filtering
    # Much like the previous filters, we start by creating a filter object:
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    # pcl.save(cloud_filtered, "table_scene_inliers.pcd")
    #
    # outlier_filter.set_negative(True)
    # pcl.save(outlier_filter.filter(), "table_scene_outliers.pcd")

    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size
    # Experiment and find the appropriate size!
    LEAF_SIZE = .01

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    # filename = 'voxel_downsampled.pcd'
    # pcl.save(cloud_filtered, filename)

    # PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.4
    axis_max = 3.
    passthrough.set_filter_limits(axis_min, axis_max)

    passthrough = passthrough.filter().make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()
    # filename = 'pass_through_filtered.pcd'
    # pcl.save(cloud_filtered, filename)

    # RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = .01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    cloud_table = extracted_inliers
    cloud_objects = extracted_outliers

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(30)
    ec.set_MaxClusterSize(1200)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # complete this step just as is covered in capture_features.py
        # Compute histograms for the clusters
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Initialize variables
    test_scene_num = Int32()
    dropbox = {}

    class PickObject:
        def __init__(self, name, group):
            self.name = String()
            self.name.data = name
            self.group = group
            self.arm = String()
            self.arm.data = 'right' if group == 'green' else 'left'
            self.centroid = None  # tuples (x, y, z)
            self._pick_pose = None
            self._place_pose = None
            self._yaml_dict = None

        @property
        def pick_pose(self):
            if self._pick_pose is None:
                self._pick_pose = Pose()
                # rospy.loginfo("pick_pose {} centroid: {}".format(self.name.data, self.centroid))
                pos = [np.asscalar(x) for x in self.centroid] if self.centroid is not None else [0,0,0]
                self._pick_pose.position.x = pos[0]
                self._pick_pose.position.y = pos[1]
                self._pick_pose.position.z = pos[2]

            return self._pick_pose

        @property
        def place_pose(self):
            if self._place_pose is None:
                self._place_pose = Pose()
                # rospy.loginfo("place_pose {} arm: {}".format(self.name.data, self.arm.data))
                pos = dropbox[self.arm.data]
                self._place_pose.position.x = pos[0]
                self._place_pose.position.y = pos[1]
                self._place_pose.position.z = pos[2]

            return self._place_pose

        @property
        def yaml_dict(self):
            if self._yaml_dict is None:
                self._yaml_dict = make_yaml_dict(test_scene_num, self.arm,
                                                 self.name, self.pick_pose,
                                                 self.place_pose)
            return self._yaml_dict

        def __str__(self):
            return "pick {} {} {} {} {} {}".format(self.name.data, self.group,
                                                   self.arm.data, self.centroid,
                                                   self.pick_pose.position,
                                                   self.place_pose.position)


    # Parse parameters into individual variables
    test_scene_num.data = 3
    rospy.loginfo('test_scene_num: {}'.format(test_scene_num.data))
    # create the pick list
    pick_list = []
    for i in range(len(object_list_param)):
        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        pick_list.append(PickObject(object_name, object_group))

    rospy.loginfo("pick_list summary {}".format([pl.name.data for pl in pick_list]))

    # load dropbox dictionary
    for i in range(len(dropbox_param)):
        name = dropbox_param[i]['name']
        position = dropbox_param[i]['position']
        dropbox[name] = position

    rospy.loginfo("dropbox {}".format(dropbox))

    # pre-calculate the centroids of all detected labels and assign to pick list
    for o in object_list:
        points_arr = ros_to_pcl(o.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        rospy.loginfo('object label {} centroid {}'.format(o.label, centroid))

        assigned = False
        for po in pick_list:
            if po.name.data == o.label and po.centroid is None:
                po.centroid = centroid
                assigned = True
                rospy.loginfo('Pick object name {} assigned centroid {}'.format(po.name.data, po.centroid))
                break

        if not assigned:
            rospy.loginfo('Detected object {} isnt in picklist or has already been assigned'.format(o.label))

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for pick_object in pick_list:
        rospy.loginfo(pick_object)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            rospy.loginfo('pick_object.pick_pose {}'.format(pick_object.pick_pose))
            rospy.loginfo('pick_object.place_pose {}'.format(pick_object.place_pose))
            # resp = pick_place_routine(test_scene_num, pick_object.name,
            #                           pick_object.arm, pick_object.pick_pose,
            #                           pick_object.place_pose)
            #
            # print ("Response: ", resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    #  Output your request parameters into output yaml file
    yaml_filename = '/tmp/output_{}.yaml'.format(test_scene_num.data)
    yaml_dict_list = [po.yaml_dict for po in pick_list]
    rospy.loginfo("writing {}".format(yaml_filename))

    send_to_yaml(yaml_filename, yaml_dict_list)

if __name__ == '__main__':


    # ROS node initialization
    rospy.init_node('perception_pipeline', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
