#!/usr/bin/env python
import sys

import rospy
import copy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan, Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Twist, Point
from referee.srv import TeamReady, SendColor, SendDimensions

import cv2
from cv_bridge import CvBridge, CvBridgeError

from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d

from nav_msgs.msg import Odometry
import numpy as np
import time
import math


def detect_objects(current_pcd, labels):
    """Use this to convert RGB to LAB
    """

    current_pcd_colors_array = np.asarray(current_pcd.colors)

    if current_pcd_colors_array is None or len(current_pcd_colors_array) == 0:
        return

    current_pcd_colors_array_hsv = cv2.cvtColor(current_pcd_colors_array.reshape(-1, 1, 3).astype(np.float32),
                                                cv2.COLOR_RGB2HSV).reshape(-1, 3)

    unique_labels = np.unique(labels)

    enemy_robot = []
    enemy_pucks = []
    my_pucks = []
    border = []

    border_labels = []
    enemy_pucks_labels = []
    enemy_robot_labels = []
    my_pucks_labels = []

    for i, label in enumerate(unique_labels):
        current_pcd_of_label = current_pcd_colors_array_hsv[labels == label]
        mean_hsv_color = np.median(current_pcd_of_label, axis=0)

        indices = np.where(labels == label)[0]
        selected_points = current_pcd.select_by_index(indices)
        if is_green_hsv(mean_hsv_color):
            border.append(selected_points)
            border_labels.extend([label] * len(selected_points.points))
        elif is_yellow_hsv(mean_hsv_color):
            enemy_pucks.append(selected_points)
            enemy_pucks_labels.extend([label] * len(selected_points.points))
        elif  is_red_hsv(mean_hsv_color):
            enemy_robot.append(selected_points)
            enemy_robot_labels.extend([label] * len(selected_points.points))
        elif is_blue_hsv(mean_hsv_color):
            my_pucks.append(selected_points)
            my_pucks_labels.extend([label] * len(selected_points.points))

    border_pcd = o3d.geometry.PointCloud()
    if border:
        border_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points for pcd in border]))
        border_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors for pcd in border]))

    enemy_pucks_pcd = o3d.geometry.PointCloud()
    if enemy_pucks:
        enemy_pucks_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points for pcd in enemy_pucks]))
        enemy_pucks_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors for pcd in enemy_pucks]))

    enemy_robot_pcd = o3d.geometry.PointCloud()
    if enemy_robot:
        enemy_robot_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points for pcd in enemy_robot]))
        enemy_robot_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors for pcd in enemy_robot]))

    my_pucks_pcd = o3d.geometry.PointCloud()
    if my_pucks:
        my_pucks_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points for pcd in my_pucks]))
        my_pucks_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors for pcd in my_pucks]))

    # rospy.loginfo(f"border {border} \ncurrent_pcd: {current_pcd}")
    # o3d.visualization.draw_geometries([current_pcd], window_name="Ground Plane Segmentation")
    # o3d.visualization.draw_geometries([border_pcd], window_name="Ground Plane Segmentation")
    # o3d.visualization.draw_geometries([enemy_pucks_pcd], window_name="Ground Plane Segmentation")
    # o3d.visualization.draw_geometries([enemy_robot_pcd], window_name="Ground Plane Segmentation")
    # o3d.visualization.draw_geometries([my_pucks_pcd], window_name="Ground Plane Segmentation")

    # region
    unique_labels = np.unique(border_labels)
    if len(unique_labels) < 3:
        play_node.set_velocities(0, 0.5)
        rospy.sleep(1)
        play_node.set_velocities(0, 0)
        return None, None
    centroids = []
    distances_per_pole = {}
    distances_per_cluster = {}

    for label in unique_labels:
        cluster_points = np.asarray(border_pcd.points)[border_labels == label]
        centroid = (np.mean(cluster_points, axis=0))
        distance_from_origin = np.linalg.norm(centroid)
        distances_per_pole[label] = centroid

    distances_per_pole_sorted = sorted(distances_per_pole.items(), key=lambda x: np.linalg.norm(x[1]))
    for i in range(len(distances_per_pole_sorted)-1):

        current_label, current_pole_centroid = distances_per_pole_sorted[i]
        next_label, next_pole_centroid = distances_per_pole_sorted[i + 1]

        cluster_pair = (current_label, next_label)

        distance = np.linalg.norm(np.asarray([next_pole_centroid[0], next_pole_centroid[1]])
                                  - np.asarray([current_pole_centroid[0], current_pole_centroid[1]]))
        distances_per_cluster[cluster_pair] = distance

    game_field = calculate_dimensions(distances_per_cluster, distances_per_pole_sorted)

    rospy.loginfo(f"game_field x: {game_field['x']} game_field y: {game_field['y']}")

    return game_field

def calculate_dimensions(distances, distances_per_pole_sorted):
    percentages = [0.10, 0.15, 0.25]
    distances_sorted = sorted(distances.items(), key=lambda x:x[1])

    cluster_pair, min_distance = distances_sorted[0]
    cluster1, cluster2 = cluster_pair

    matching_key = None
    for key in distances.keys():
        if key[1] == cluster1:
            matching_key = key[0]
            break
        elif key[0] == cluster2:
            matching_key = key[1]
            break
    distances_per_pole_sorted_dict = dict(distances_per_pole_sorted)

    for percentage1 in percentages:
        reference_value = min_distance / percentage1
        for lable, dist in distances_sorted:
            for percentage in percentages[1:]:
                tolerance = 0.03
                expected_value = reference_value * percentage
                if abs(dist - expected_value) <= tolerance:

                    min_distance_poles_coordinates = {'pol1': distances_per_pole_sorted_dict[cluster1],
                                                      'pol2': distances_per_pole_sorted_dict[cluster2],
                                                      'pol3': distances_per_pole_sorted_dict[matching_key],
                                                      'percentage_1_2': percentage1,
                                                      'percentage_2_3': percentage}
                    x = reference_value
                    y = reference_value/5 * 3
                    game_field = {'x': x * 100, 'y': y * 100}

                    localize_robot(game_field, min_distance_poles_coordinates)
                    return game_field



#udom
# add rotation if there are no pols
def localize_robot(game_field, min_distance_poles_coordinates):

    if np.linalg.norm(min_distance_poles_coordinates['pol1']) < np.linalg.norm(min_distance_poles_coordinates['pol2']):
        vector_to_middle_pole = -min_distance_poles_coordinates['pol1']
        vector_between_poles = min_distance_poles_coordinates['pol2'] - min_distance_poles_coordinates['pol1']
    else:
        vector_to_middle_pole = -min_distance_poles_coordinates['pol2']
        vector_between_poles = min_distance_poles_coordinates['pol1'] - min_distance_poles_coordinates['pol2']

    dot_product = np.dot(vector_between_poles, vector_to_middle_pole)

    magnitude1 = np.linalg.norm(vector_to_middle_pole)
    magnitude2 = np.linalg.norm(vector_between_poles)

    cosine_theta = dot_product / (magnitude1 * magnitude2)
    angle_rad_between_vectors = np.arccos(cosine_theta)
    angle_between_vectors = np.degrees(angle_rad_between_vectors)

    if angle_between_vectors > 90:
        angle_to_pole = 180 - angle_between_vectors
    else:
        angle_to_pole = angle_between_vectors

    y = np.sin(np.deg2rad(angle_to_pole)) * magnitude1 * 100
    x = np.cos(np.deg2rad(angle_to_pole)) * magnitude1 * 100
    delta_x = 0
    if np.linalg.norm(min_distance_poles_coordinates['pol3']) < magnitude1:
        p = min_distance_poles_coordinates['percentage_1_2']
        percentage = 1
        if p == 0.10:
            percentage = 0.9
        elif p == 0.15:
            percentage = 0.75
        elif p == 0.25:
            percentage = 0.5
        delta_x = percentage * game_field['x']
    else:
        p = min_distance_poles_coordinates['percentage_1_2']
        if p != 0.10:
            delta_x = p * game_field['x']

    distance_to_robot = {'x': (x - delta_x),'y': y}
    rospy.loginfo(f"distance_to_robot {distance_to_robot}")


def mark_objects_of_interest(current_pcd, labels, mean_lab_colors):
    """ Use to visualize detected objects
    """

    max_label = labels.max() + 1
    bounding_boxes = []

    for i in range(max_label):
        border_color = is_object_of_interest(mean_lab_colors[i])

        if not border_color:
            continue

        cluster_points = np.where(labels == i)[0]
        pcd_i = current_pcd.select_by_index(cluster_points)

        aabb = pcd_i.get_axis_aligned_bounding_box()

        aabb.color = border_color

        bounding_boxes.append(aabb)

    o3d.visualization.draw_geometries([current_pcd] + bounding_boxes)


def is_object_of_interest(lab_color):
    green = is_green_hsv(lab_color)
    yellow = is_yellow_hsv(lab_color)
    red = is_red_hsv(lab_color)
    blue = is_blue_hsv(lab_color)

    if green:
        return green
    elif yellow:
        return yellow
    elif red:
        return red
    elif blue:
        return blue
    return False

 # region check color
def is_green_hsv(HSV_color):
    H, S, V = HSV_color
    rospy.loginfo(f"h {H} \ns: {S}\nv: {V}")
    return (110 <= H <= 140)

def is_yellow_hsv(HSV_color):
    H, S, V = HSV_color
    return (40 <= H <= 65)

def is_red_hsv(HSV_color):
    H, S, V = HSV_color
    return ((0 <= H <= 20) or (340 <= H <= 360))

def is_blue_hsv(HSV_color):
    H, S, V = HSV_color
    return (210 <= H <= 255)

def is_yellow_goal_hsv(HSV_color):
    H, S, V = HSV_color
    return (40 <= H <= 65)

# todo change color
def is_blue_goal_hsv(HSV_color):
    H, S, V = HSV_color
    return (210 <= H <= 255)

def filter_ground_colors(ground_plane):
    points = np.asarray(ground_plane.points)
    colors = np.asarray(ground_plane.colors)

    ground_plane_colors_array_hsv = cv2.cvtColor(colors.reshape(-1, 1, 3).astype(np.float32),
                                                cv2.COLOR_RGB2HSV).reshape(-1, 3)
    indices_yellow_goal = np.array([is_yellow_goal_hsv(color) for color in ground_plane_colors_array_hsv])
    indices_blue_goal = np.array([is_blue_goal_hsv(color) for color in ground_plane_colors_array_hsv])

    filtered_point_cloud_yellow_goal = o3d.geometry.PointCloud()
    filtered_point_cloud_blue_goal = o3d.geometry.PointCloud()

    if np.any(indices_yellow_goal):
        filtered_points_yellow_goal = points[indices_yellow_goal]
        filtered_colors_yellow_goal = colors[indices_yellow_goal]
        filtered_point_cloud_yellow_goal.points = o3d.utility.Vector3dVector(filtered_points_yellow_goal)
        filtered_point_cloud_yellow_goal.colors = o3d.utility.Vector3dVector(filtered_colors_yellow_goal)
        yellow_goal_labels = np.array(filtered_point_cloud_yellow_goal.cluster_dbscan(eps=0.15, min_points=15, print_progress=True))
    else:
        yellow_goal_labels = np.array([])

    if np.any(indices_blue_goal):
        filtered_points_blue_goal = points[indices_blue_goal]
        filtered_colors_blue_goal = colors[indices_blue_goal]
        filtered_point_cloud_blue_goal.points = o3d.utility.Vector3dVector(filtered_points_blue_goal)
        filtered_point_cloud_blue_goal.colors = o3d.utility.Vector3dVector(filtered_colors_blue_goal)
        blue_goal_labels = np.array(filtered_point_cloud_blue_goal.cluster_dbscan(eps=0.15, min_points=15, print_progress=True))
    else:
        blue_goal_labels = np.array([])

    return (filtered_point_cloud_yellow_goal, yellow_goal_labels, filtered_point_cloud_blue_goal, blue_goal_labels)


def categorise_goals(ground, labels):
    # todo
    return
def get_distance(cluster1, cluster2):

    centroid1 = np.mean(np.asarray(cluster1.points), axis=0)
    centroid2 = np.mean(np.asarray(cluster2.points), axis=0)

    distance = np.linalg.norm(centroid1 - centroid2)
    rospy.loginfo(f"distance #{distance}")
    return distance


class PlayNode:

    def __init__(self, window_name="Kinect image"):
        """
        :param message_slop: Messages with a header.stamp within message_slop
        seconds of each other will be considered to be synchronisable
        """
        rospy.init_node("robot_node")
        rospy.loginfo("Initialised PlayNode")

        self.bridge = CvBridge()
        self.image_window = window_name
        self.image = None
        self.laser = None
        self.pcd = None
        self.camera_matrix = np.eye(3)

        velocity_topic = "cmd_vel"
        image_topic = "kinect/rgb/image_raw"
        laser_topic = "front_laser/scan"
        pointcloud_topic = "kinect/depth_registered/points"
        camerainfo_topic = "kinect/rgb/camera_info"

        if rospy.get_namespace() == "/":
            velocity_topic = "robot1/" + velocity_topic
            image_topic = "robot1/" + image_topic
            laser_topic = "robot1/" + laser_topic
            pointcloud_topic = "robot1/" + pointcloud_topic
            camerainfo_topic = "robot1/" + camerainfo_topic

        self.velocity_pub = rospy.Publisher(velocity_topic, Twist, queue_size=1000)
        self.status_pub = rospy.Publisher("/robot_status", String, queue_size=10)
        self.team_info_pub = rospy.Publisher("/team_info", String, queue_size=10)

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb)
        self.start_stop_sub = rospy.Subscriber("/ref/start_stop", Bool, self.start_stop_cb)
        self.team_color_sub = rospy.Subscriber("/ref/team_color", String, self.team_color_cb)
        
        rospy.sleep(.5)
        self.pointcloud_sub = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pointcloud_cb)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.camera_cb)
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_cb)
        
        self.game_active = False
        self.pcd_labels = None
        
        self.position = np.array([0.0, 0.0, 0.0])
        self.last_time = rospy.Time.now()
        
        self.register_team("R2D2")
        self.assign_team_color("R2D2", "blue")
        self.send_field_dimensions("R2D2", 5.0, 3.0)
        
    def register_team(self, team_name):
        rospy.loginfo(f"Registering team: {team_name}")
        rospy.wait_for_service('/TeamReady')
        try:
            team_ready_srv = rospy.ServiceProxy('/TeamReady', TeamReady)
            response = team_ready_srv(team_name)
            if response.ok:
                rospy.loginfo(f"Team {team_name} registered successfully")
            else:
                rospy.logwarn(f"Failed to register team: {team_name}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def assign_team_color(self, team_name, color):
        rospy.loginfo(f"Assigning color {color} to team {team_name}")
        rospy.wait_for_service('/SendColor')
        try:
            send_color_srv = rospy.ServiceProxy('/SendColor', SendColor)
            response = send_color_srv(team_name, color)
            if response.ok:
                rospy.loginfo(f"Team {team_name} color set to {color}")
            else:
                rospy.logwarn(f"Received corrected color {response.correctColor} for team {team_name}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def send_field_dimensions(self, team_name, width, length):
        rospy.loginfo(f"Sending dimensions for {team_name}: width={width}, length={length}")
        rospy.wait_for_service('/SendDimensions')
        try:
            send_dimension_srv = rospy.ServiceProxy('/SendDimensions', SendDimensions)
            dimensions = Point(x=width, y=length, z=0)
            response = send_dimension_srv(team_name, dimensions)
            if response.ok:
                rospy.loginfo(f"Dimensions for {team_name} set to width={response.correctDimensions.x}, length={response.correctDimensions.y}")
            else:
                rospy.logwarn(f"Failed to set dimensions for {team_name}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}") 

    def start_stop_cb(self, msg):
        """Callback function for start/stop messages."""
        rospy.loginfo(f"Received start/stop message: {msg.data}")
        self.game_active = msg.data
        
    def team_color_cb(self, msg):
        """Callback function for team color messages."""
        rospy.loginfo(f"Received team color message: {msg.data}")

    def laser_cb(self, msg):
        """This function is called every time a message is received on the
        front_laser/scan topic

        We just keep the self.laser member up to date
        """
        rospy.loginfo("Received new scan")
        self.laser = msg

    def camera_cb(self, msg):
        """This function is called every time a message is received on the
        front_camera/image_raw topic

        We convert the image to an opencv object and update the self.image member
        """
        rospy.loginfo("Received new image")

        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

    def pointcloud_cb(self, msg):
        """This function is called every time a message is received on the
        kinect/depth_registered/points topic

        We just keep the self.pcd member up to date
        """
        rospy.loginfo("Received new scan")
        self.pcd = orh.rospc_to_o3dpc(msg, remove_nans=True)
        self.segment_cluster_floor_and_objects()
        # Can be used for visualization.
        # o3d.visualization.draw_geometries([self.pcd])
        print(self.pcd)

    def camera_info_cb(self, msg):

        for i in range(3):
            for k in range(3):
                self.camera_matrix[i][k] = msg.K[i * 3 + k]
                print(i * 3 + k)
                
        self.image_height = msg.height
        self.image_width = msg.width
        self.annotated_img = np.zeros(shape=(self.image_height, self.image_width))
        self.camera_info_sub.unregister()
        print("Got Camera Parameters")
        print(self.camera_matrix)

    def show_img(self):
        """Need to do this separately because otherwise it can freeze
        """
        if self.image is not None:
            cv2.imshow(self.image_window, self.image)
            cv2.waitKey(10)
        else:
            rospy.loginfo("No image to show yet")

    def set_velocities(self, linear, angular):
        """Use this to set linear and angular velocities
        """
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.velocity_pub.publish(msg)

    def segment_cluster_floor_and_objects(self):

        if self.pcd is None:
            rospy.loginfo("No point cloud data available (segment_floor_and_objects)")
            return

        # region down sample
        current_pcd = self.pcd.voxel_down_sample(voxel_size=0.027)

        # region plan segmentation
        plane_model, inliers = current_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
        ground_plane = current_pcd.select_by_index(inliers)
        current_pcd = current_pcd.select_by_index(inliers, invert=True)
        # normal vector of the plane
        normal = plane_model[:3]

        # Function for ignoring walls/tables etc
        # Check if the normal vector is aligned with the z-axis
        if abs(normal[0]) < 0.1 and abs(normal[1]) < 0.1 and normal[2] > 0.9:
            # normal[0] = x ; normal[1] = y ; normal[2] = z
            # Proceed with processing the ground plane
            # Perform clustering and object detection
            labels = np.array(current_pcd.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
            detect_objects(current_pcd, labels)
        else:
            rospy.loginfo("Plane detected is not the floor. Ignoring.")

        yellow_goal_pcd, yellow_goal_labels, blue_goal_pcd, blue_goal_labels = filter_ground_colors(ground_plane)


        # region detect objects
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            current_pcd_labels = np.array(current_pcd.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
        if len(current_pcd_labels) == 0:
            print("no cluster found")
            # return
        # endregion


    #result visualization with rostopic echo /robot1/odom, in terminal: position = current location(x,y)/orientation(z,w), twist = current speed 
    def odom_cb(self, msg):  
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        dx = msg.twist.twist.linear.x * dt
        dy = msg.twist.twist.linear.y * dt
        dtheta = msg.twist.twist.angular.z * dt

        self.position[0] += dx * np.cos(self.position[2]) - dy * np.sin(self.position[2])
        self.position[1] += dx * np.sin(self.position[2]) + dy * np.cos(self.position[2])
        self.position[2] += dtheta

        rospy.loginfo(f"Updated position: x={self.position[0]:.2f}, y={self.position[1]:.2f}, theta={self.position[2]:.2f}")
        
def initialize_game(play_node):
    rospy.loginfo("Initializing game by spinning to detect poles")
    loop_rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        play_node.set_velocities(0, 0.1)
        
        if play_node.pcd is not None and play_node.pcd_labels is not None:
            game_field = detect_objects(play_node.pcd, play_node.pcd_labels)
            
            if game_field is not None:
                rospy.loginfo("Poles detected, stopping spin")
                play_node.set_velocities(0, 0)
                break

        loop_rate.sleep()

    rospy.loginfo("Game initialization complete")

if __name__ == '__main__':
    play_node = PlayNode()
    rospy.loginfo("Starting hockey robot game")
    
    initialize_game(play_node)
    
    loop_rate = rospy.Rate(10)  
    while not rospy.is_shutdown():

        loop_rate.sleep()

