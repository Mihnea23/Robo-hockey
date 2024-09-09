Plane Segmentation:
Objective: Segment the ground plane from other objects using the RANSAC algorithm.

RANSAC Algorithm: This algorithm is used to estimate the ground plane's parameters. 
It distinguishes the ground from other objects based on the normal vector.
If the vector aligns with the z-axis and falls within specific orientation thresholds, it is considered the floor.

Key Libraries:
Open3D: Used for 3D data processing and visualization.
OpenCV (cv2): Used for image processing and color manipulation.

Function Descriptions (plane segmentation):
1. segment_cluster_floor_and_objects:
This function separates the ground plane from the environment by applying fitting algorithms and filters to exclude non-floor planes.
2. pointcloud_cb:
Triggered when a new point cloud message is received. It performs voxel downsampling to create a uniformly reduced point cloud from the input data.

Object Clustering and Color Detection:
After separating the ground plane, the next step is to cluster objects using the cluster_dbscan function from Open3D.
Once objects are clustered, color differentiation is applied:
RGB color format is converted to LAB format (using OpenCV) because LAB is more effective for color detection.
Median colors for each object are computed and compared to predefined LAB color ranges to identify specific objects.
Finally, the detected objects are visualized with bounding boxes around them.

Field Dimension Detection:
The detection of field dimensions is based on clustered boundary objects.
The process involves:
1. Calculating the center of each object to determine vectors between the robot and the center.
2. Measuring distances between poles and using the smallest distances to establish relationships between poles.

To calculate field dimensions:
Distances between poles are sorted.
A minimum of three poles is required to calculate the field width and length using percentage-based comparisons of the distances.
The width is derived by dividing the smallest distance by the identified percentage.
The length is calculated using the relationship: Length = (Width * 5) / 3. (based on the known field parameters)

Robot Position and Boundary Distance Calculation:
A function is implemented to calculate the angle between two vectors. This helps determine the robot's distance from the field's boundaries.
The process involves:
Computing the angle between the vector from the robot to the nearest pole and the field's border.
Using this angle and robot coordinates to determine distances between poles and the robot relative to the field's coordinate system.
Horizontal Position Calculation
The robot's horizontal location (x-component) is determined using vectors:
From the robot to the center of objects.
Between poles to compare distance percentages along the y-axis.
Three poles are considered at a time for efficient location detection.
Using these vectors, a reference angle is calculated with the cosine law, helping compute the horizontal component of the robot’s position.
Visualization and Pose Estimation

PoseStamped Library is used for better visualization via Rviz (a visualization tool for ROS).
The robot's current position, orientation, and speed are tracked using odometry.
The odom_cb function integrates data over time as the robot moves, calculating dx, dy, and theta (rotation angle).
Visualization is done using the command echo /robot1/odom in the terminal.

Referee Node and Game Server Communication:
A new script ref.py contains a referee node that allows the robot to communicate with the game server.
The node handles commands such as TeamReady, SendColor, SendDimensions, and control commands like start, stop.
Adjustments were made to the initialization function in the main player node to set up team details and dimensions dynamically,
using the calculated field dimensions instead of predefined values.
