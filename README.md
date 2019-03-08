# **Programming a Real Self-Driving Car**

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

[//]: # (Image References)

[image1]: ./imgs/demo.gif "demo"
[image2]: ./imgs/final-project-ros-graph-v2.png "final-project-ros-graph-v2"
[image3]: ./imgs/tl-detector-ros-graph.png "tl-detector-ros-graph"
[image4]: ./imgs/waypoint-updater-ros-graph.png "waypoint-updater-ros-graph"
[image5]: ./imgs/dbw-node-ros-graph.png "dbw-node-ros-graph"
[image6]: ./imgs/obj_det.png "obj_det"
[image7]: ./imgs/waypoint.png "waypoint"

![alt text][image1]

---
### Team Member

| Name  | Udacity account email |
|-------|-----------------------|
|Tianqi Ye| ye.tianqi1900@gmail.com|
|       |                       |
| FC Su | dragon7.fc@gmail.com  |
| Zhong | 546764887@qq.com      |
|Fujing Xie|emiliexiaoxiexie@hotmail.com                       |



---
### System Architecture Diagram

![alt text][image2]

### Code Structure

* Traffic Light Detection Node

    - (path_to_project_repo)/ros/src/tl_detector/

        This package contains the traffic light detection node: `tl_detector.py`. This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint` topic.

        The `/current_pose` topic provides the vehicle's current position, and `/base_waypoints` provides a complete list of waypoints the car will be following.

        Traffic light detection should take place within `tl_detector.py`, whereas traffic light classification should take place within `../tl_detector/light_classification_model/tl_classfier.py`.

        We applied Tensorflow to detect traffic light. The real-time object detection model we used is [SSD_Mobilenet 11.6.17 version](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz).

        After successfully detected traffic light, we converted RGB to HSV color space to identify red/green/yellow light.

        ![alt text][image3]

        ![MobileNets Graphic](https://github.com/tensorflow/models/raw/master/research/slim/nets/mobilenet_v1.png)

    ![alt text][image6]

* Waypoint Updater Node

    - (path_to_project_repo)/ros/src/waypoint_updater/

        This package contains the waypoint updater node: `waypoint_updater.py`. The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node will subscribe to the `/base_waypoints`, `/current_pose`, `/obstacle_waypoint`, and `/traffic_waypoint` topics, and publish a list of waypoints ahead of the car with target velocities to the `/final_waypoints` topic.
        The waypoint updater node serves the following functions:
        * A KD-Tree algorithm was used to search for the nearest waypoint, and bring down the search time to O(logn). 
        * A vector products function was used to help the vehicle detect if the nearest waypoint is behind the vehicle.
        * A decelerate waypoint will be generated if a red traffic light is detected.
        
        The pure pursuit line-follow strategy was used after the waypoint gerneation. A DBW control node(details in next section) was used to help the vehicle keep tracking of the target waypoint. The steering angle is generated based on the position next target waypoint for each loop, using the vehicle kinematics and the trun curvature calculation.

        ![alt text][image4]

    ![alt text][image7]

* DBW Node

    - (path_to_project_repo)/ros/src/twist_controller/

        Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package contains the files that are responsible for control of the vehicle: the node `dbw_node.py` and the file `twist_controller.py`, along with a pid and lowpass filter that you can use in your implementation. The `dbw_node` subscribes to the `/current_velocity` topic along with the `/twist_cmd` topic to receive target linear and angular velocities. Additionally, this node will subscribe to `/vehicle/dbw_enabled`, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics.
        A brake of 400Nm is applied to make the vehicle hold staionary and it will dynamically change with the vehicle current speed and vehicle mass(v/s^2 * kg = N*m) during a deceleration situation. 

        ![alt text][image5]

---
Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```
__NOTE__: **The docker will run fail with camera enable**.


### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
