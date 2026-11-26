### 各传感器话题频率

1. 小车odom：50hz
2. imu：100hz
3. zed相机：15hz

### 网络优先级配置

```bash
sudo nmcli connection modify "WiFi-Network" ipv4.route-metric 100
sudo nmcli connection modify "Wired connection 1" ipv4.route-metric 200

nmcli connection down "WiFi-Network" && nmcli connection up "WiFi-Network"
nmcli connection down "Wired connection 1" && nmcli connection up "Wired connection 1"

sudo systemctl restart NetworkManager
```

### 自动安装所有依赖

```
rosdep install --from-paths src --ignore-src -r -y
```



### 一、conda与ros环境

退出conda环境恢复ros环境代码

```
conda deactivate
```

启动禁止conda环境默认ros

```
sudo vim ~/.condarc
auto_activate_base: false
```

### 二、ROS工作空间问题source

把*source ./devel/setup.bash*添加到*.bashrc*文件

```
gedit ~/.bashrc
```

在文件末尾添加以下内容

```
source ~/你的工作空间路径/devel/setup.bash
```

保存并关闭文件，然后运行以下命令使修改立即生效

```
source ~/.bashrc
```

### 三、查看tf树

```
rosrun rqt_tf_tree rqt_tf_tree    查看TF树
rosrun rqt_reconfigure rqt_reconfigure
```

### 四、查看话题流向

```
rosrun rqt_graph rqt_graph
```

```
rostopic ech /scan --noarr
```

查看/scan话题数据和对应其对应的坐标系（frame_id）

找对应的ros包

```
rospack find 包名
```



### 五、坐标系变换

```
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 lslidar base_link
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_footprint base_link

rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 front_left_wheel_link map
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 front_right_wheel_link map
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 rear_left_wheel_link map
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 rear_right_wheel_link map
```

可在.launch文件中添加

```
<node pkg="tf2_ros" type="static_transform_publisher" name="hokuyo_tf_broadcaster" 
        args="0 0 0 0 0 0  base_link hokuyo_link" />
```

这里的参数 `0 0 0 0 0 0` 分别代表 `x y z yaw pitch roll`

## 配置文件

1. scout/scout_gazebo_sim/launch/scout_mini_playpen.launch 

   -  加载虚拟世界文件
   - 调用spawn_scout_mini.launch

2. scout/scout_gazebo_sim/launch/spawn_scout_mini.launch

   - **参数化配置**：机器人初始位姿、传感器选型 / 安装位置、命名空间等。
   - **模型组装**：调用scout_mini_stock.launch文件构建完整 URDF 模型（含底盘 + 传感器）。
   - **仿真集成**：将 URDF 加载到 Gazebo，启动运动控制器和状态发布器。

3. scout/scout_description/launch/scout_mini_stock.launch

   - 配置 Scout Mini 机器人的传感器到base_link的变换
   - 调用mini.xacro

4. scout/scout_description/urdf/mini.xacro

   - 机器人的核心模型定义文件
   - 通过条件判断动态加载不同传感器
   - 调用imu.xacro、gps.xacro、camera.xacro、hokuyo_2d.xacro

5. 

   # scout_mini_stock.launch

## 启动各节点
0.调试
```
避免太乱，文件复制到~/hjd下，每次开启新终端时输入下面一行指令
source ~/hjd/catkin_ws/devel/setup.bash
启动各坐标系tf发布
roslaunch scout_description display_scout_mini.launch
启动下面这个就好 发布坐标，启动雷达，相机，EKF等
roslaunch scout_bringup scout_robot_base_nav_test.launch
还需要启动IMU（单开窗口）
cd ~/AIgoT1
source ./devel/setup.bash
roslaunch algot1_base algo-algot1.launch

航点巡航
cd /home/ouc/scout_nav/src/nav
source ./devel/setup.bash
rosrun waypoint_recorder patrol_waypoints

yolo实时识别
python3 ~/scout_nav/src/nav/src/yolo_detector/scripts/yolo_zed_node.py \
  _image_topic:=/zed2i/zed_node/rgb/image_rect_color \
  _weights:=/home/ouc/scout_nav/src/nav/src/yolo_detector/weights/yolov5s.pt \
  _img_size:=320 \
  _conf:=0.4
查看
rqt_image_view /yolo/image
```
启动rtab-map
```
roslaunch car_launch rtabmap_mapping.launch
```

1. 启动基础节点

   ```
   roslaunch scout_bringup scout_mini_robot_base.launch
   roslaunch scout_bringup scout_base.launch
   roslaunch scout_bringup scout_robot_base.launch
   ```

2. 启动键盘远程操作节点

   ```
   roslaunch scout_bringup scout_teleop_keyboard.launch
   ```

   -  u, i, o：控制机器人的左选、前进和右旋（u向左旋转，i 向前，o 向右旋转）。

   - j, k, l：控制机器人向左移动、停止、向右移动。
   - q/z：增加/减少最大速度（10%）。
   - w/x：增加/减少仅线性速度（10%）。
   - e/c：增加/减少仅角速度（10%）。
   - 任何其他键：停止机器人的所有运动。
   - 使用 CTRL-C 可以退出控制。
   - 

3. 启动雷达

   ```
   roslaunch lslidar_driver lslidar_N301.launch
   ```

4. 启动IMU

   ```
   cd ~/AIgoT1
   source ./devel/setup.bash
   roslaunch algot1_base algo-algot1.launch
   ```

5. 启动相机

   ```
   <!-- roslaunch zed_display_rviz display_zed.launch -->
   roslaunch zed_wrapper zed2i.launch
   /usr/local/zed/tools/ZED_Explorer
   ```

6. 启动卡尔曼滤波器

   ```
   roslaunch nav_demo ekf_cmd.launch
   roslaunch nav_demo ekf.launch
   ```
   
7. SLAM建图

   ```
   roslaunch nav_demo gmapping_slam.launch
   
   cd ~/2D-SLAM
   source ./devel_isolated/setup.bash
   roslaunch cartographer_ros lx_rs16_2d_outdoor_real_time.launch
   ```
   
   
   
8. 保存地图

   ```
   roslaunch nav_demo gmapping_save.launch
   ```

9. 启动amcl

   ```
   roslaunch nav_demo test_amcl.launch
   ```

10. 启动导航

    ```
    roslaunch nav_demo nav_test.launch
    roslaunch nav_demo auto_slam.launch
    ```
    11.自主探索
    roslaunch explore_lite explore_costmap.launch
    
    12.地图覆盖导航
    
    

使用方式一：只生成点，不执行

这个适合先调试。

```
rosrun waypoint_recorder coverage_patrol_node \
  _map_yaml:=/home/ouc/maps/lab_map.yaml \
  _output_file:=/home/ouc/scout_nav/src/nav/src/waypoint_recorder/param/waypoints.yaml \
  _stripe_step:=0.6 \
  _inflation_radius:=0.35 \
  _side_margin:=0.25 \
  _save_yaml:=true \
  _send_direct:=false
```

然后查看：

```
cat /home/ouc/scout_nav/src/nav/src/waypoint_recorder/param/waypoints.yaml
```

使用方式二：生成后直接发送给 move_base

前提是你已经启动了：

```
底盘
雷达
map_server
AMCL
move_base
TF
```

也就是要确保：

```
rostopic list | grep cmd_vel
rostopic list | grep scan
rostopic list | grep map
rosnode list | grep move_base
```

然后运行：

```
rosrun waypoint_recorder coverage_patrol_node \
  _map_yaml:=/home/ouc/scout_nav/src/nav/src/nav_demo/maps/lab_side_map.yaml \
  _output_file:=/home/ouc/scout_nav/src/nav/src/waypoint_recorder/param/waypoints_lab_side_map.yaml \
  _stripe_step:=0.6 \
  _inflation_radius:=0.35 \
  _side_margin:=0.25 \
  _save_yaml:=true \
  _send_direct:=true \
  _goal_timeout:=120.0 \
  _wait_time:=2.0 \
  _loop:=false
```

这个会：

```
读取地图
生成覆盖点
保存 YAML
等待 move_base
依次发送目标点
```

使用方式三：不保存 YAML，直接执行

```
rosrun waypoint_recorder coverage_patrol_node \
  _map_yaml:=/home/ouc/maps/lab_map.yaml \
  _output_file:=/tmp/coverage_waypoints.yaml \
  _stripe_step:=0.6 \
  _inflation_radius:=0.35 \
  _side_margin:=0.25 \
  _save_yaml:=false \
  _send_direct:=true \
  _goal_timeout:=120.0 \
  _wait_time:=2.0 \
  _loop:=false
```

参数怎么调

如果生成点太多：

```
_stripe_step 调大，比如 0.8
```

如果覆盖不够密：

```
_stripe_step 调小，比如 0.4
```

如果点太靠墙，小车导航失败：

```
_inflation_radius 调大，比如 0.45
_side_margin 调大，比如 0.35
```

如果想循环巡航：

```
_loop:=true
```

你现在的完整流程

应该是：

```
# 1. 启动地图、定位、导航
roslaunch scout_bringup scout_robot_base_nav_test.launch
roslaunch nav_demo nav_test.launch

# 2. 启动覆盖巡航节点
rosrun waypoint_recorder coverage_patrol_node \
  _map_yaml:=/home/ouc/scout_nav/src/nav/src/nav_demo/maps/lab_side_map.yaml \
  _output_file:=/home/ouc/scout_nav/src/nav/src/waypoint_recorder/param/waypoints_lab_side_map.yaml \
  _stripe_step:=0.6 \
  _inflation_radius:=0.35 \
  _side_margin:=0.25 \
  _save_yaml:=true \
  _send_direct:=true
```

ZED修改

1. zed-ros/zed-ros-wrapper/zed_wrapper/urdf/zed_macro.urdf.xacro
   {name}_camera_center修改为camera_depth_frame
2. 增加odomFlag：是否输出相机odom->baselink



## rosbag

1. 记录数据

   ```
   rosbag record -a 
   ```

2. 查看记录数据

   ```
   rosbag info data_name.bag
   ```

3. 播放数据

   ```
   rosbag play data_name.bag
   ```

   

## cartographer

1. 进入目录

   ```
   cd ~/2D-SLAM/src/cartographer_detailed_comments_ws/
   ```

2. 编译

   ```
   catkin_make_isolated --install --use-ninja
   source install_isolated/setup.bash
   ```

3. 运行

   ```
   roslaunch ~/2D-SLAM/src/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/launch/cartographer_baitai.launch
   ```

## 仿真-显示不同处

### 一、xacro文件

1. gazebo的urdf或xacro文件中必须设置碰撞（多出<collision>）
2. gazebo的urdf或xacro文件中使用在head.xacro中封装过的inertial标签

## ROS学习

### 一、通讯机制

```
rosnode list == 列出所有活动节点
rosnode info /话题 == 列出该话题的发出者、接收者

rostopic echo 话题 == 打印消息信息到屏幕
rostopic list == 显示所有话题
rostopic info 话题 == 列出该话题的类型、发出者和接收者
```

查看节点通讯

```
rosrun rqt_graph rqt_graph == 显示不同结点以及通讯链接的话题
rostopic info 话题 == 列出该话题的类型、发出者和接收者
rosmsg show 话题 == 显示话题消息的格式
```

### 二、launch文件

<!-- 这是注释的格式 -->

1. launcuh文件标签之node

   ```
   <node pkg="turtlesim" type="turtlesim_node"     name="myTurtle" output="screen" />
   ```

   - pkg="包名" ：节点所属的包

   - type="nodeType"：节点类型(与之相同名称的可执行文件)
   - name="nodeName"：节点名称(在 ROS 网络拓扑中节点的名称)

2. launcuh文件标签之include

   ```
   <include file = "$(find launch01_basic)/launch/start.launch">
   ```

   代码作用：找到launch01_basic功能包下launch目录的start.launch并启动

   - file="$(find 包名)/xxx/xxx.launch"： 要包含的文件路径
   - ns="xxx" (可选)： 在指定命名空间导入文件

3. launcuh文件标签之remap

   ```
   <node pkg="turtlesim" type="turtlesim_node"     name="myTurtle" output="screen" >
   	<remap from="/turtlel/cmd_vel" to="/cmd_vel">
   </node>
   ```

   代码作用：把turtlesim_node节点中的<font color='cornflowerblue'>turtlel/cmd_vel</font>话题修改为<font color='cornflowerblue'>cmd_vel</font>

   - from="xxx"：原始话题名称
   - to="yyy"：目标名称

4. launcuh文件标签之param

   ```
   <launch>
       <!-- 格式1：launch下，node外-->
       <param name="param_A" type="int" value="100">
   
       <!-- 格式2：node下-->
       <node pkg="turtlesim" type="turtlesim_node"     name="myTurtle" output="screen">
           <remap from="/turtlel/cmd_vel" to="/cmd_vel">
           <param name="param_B" type="double" value="3.14">
       </node>
   </launch>
   ```

   node外的参数名叫param_A，node内的参数名叫myTurtle/param_B

   - name="命名空间/参数名"

     参数名称，可以包含命名空间

   - value="xxx" (可选)

     定义参数值，如果此处省略，必须指定外部文件作为参数源

   - type="str | int | double | bool | yaml" (可选)

     指定参数类型，如果未指定，roslaunch 会尝试确定参数类型，规则如下:

     - 如果包含 '.' 的数字解析未浮点型，否则为整型
     - "true" 和 "false" 是 bool 值(不区分大小写)
     - 其他是字符串

5. launcuh文件标签之rosparam

   ```
   <launch>
       <!-- 格式1：launch下，node外-->
       <rosparam command="load" file="$(find launch01_basic)/launch/params.yaml">
   
       <!-- 格式2：node下-->
       <node pkg="turtlesim" type="turtlesim_node"     name="myTurtle" output="screen">
           <remap from="/turtlel/cmd_vel" to="/cmd_vel">
           <rosparam command="load" file="$(find launch01_basic)/launch/params.yaml">
       </node>
   </launch>
   ```

   代码功能：从 YAML 文件导入参数（参数已有名字），rosparam在node内和外的差别与前面一样（在node内名字前面会多myTurtle）

   - command="load | dump | delete" (可选，默认 load)

     加载、导出或删除参数

   - file="$(find xxxxx)/xxx/yyy...."

     加载或导出到的 yaml 文件

   - param="参数名称"

   - ns="命名空间" (可选)

6. launcuh文件标签之group

   `<group>`标签可以对节点分组，具有 ns 属性，可以让节点归属某个命名空间。避免因为同名后启动的节点把前面启动的给干掉。

   - ns="名称空间" (可选)

   - clear_params="true | false" (可选)

     启动前，是否删除组名称空间的所有参数(慎用....此功能危险)

7. launcuh文件标签之arg

   `<arg>`标签是用于动态传参，类似于函数的参数，可以增强launch文件的灵活性

   ```
   <launch>
       <arg name="car_length" default ="0.5"/>
       <param name="param" value="$(arg car_length)" />
   </launch>
   ```

   - name="参数名称"

   - default="默认值" (可选)

   - value="数值" (可选)

     不可以与 default 并存

   - doc="描述"

     参数说明

### 三、ROS常用组件

#### 3.1TF坐标变换

##### 3.1.1坐标msg消息

geometry_msgs/TransformStamped

```
std_msgs/Header header                     #头信息
  uint32 seq                                #|-- 序列号
  time stamp                                #|-- 时间戳
  string frame_id                            #|-- 坐标 ID
string child_frame_id                    #子坐标系的 id
geometry_msgs/Transform transform        #坐标信息
  geometry_msgs/Vector3 translation        #偏移量
    float64 x                                #|-- X 方向的偏移量
    float64 y                                #|-- Y 方向的偏移量
    float64 z                                #|-- Z 方向上的偏移量
  geometry_msgs/Quaternion rotation        #四元数用于表示坐标的相对姿态
    float64 x                                
    float64 y                                
    float64 z                                
    float64 w
```

代码描述：child_frame_id坐标系相对于frame_id坐标系的位姿

geometry_msgs/PointStamped

```
std_msgs/Header header                    #头
  uint32 seq                                #|-- 序号
  time stamp                                #|-- 时间戳
  string frame_id                            #|-- 所属坐标系的 id
geometry_msgs/Point point                #点坐标
  float64 x                                    #|-- x y z 坐标
  float64 y
  float64 z
```

代码描述：坐标点是以frame_id坐标系生成的

##### 3.1.2静态坐标变换

<font color='red'>所谓静态坐标变换，是指两个坐标系之间的相对位置是固定的。</font>

现实硬件各个到baselink的位置

| frame_id | child_frame-id | x y z  yaw pitch roll  |
| -------- | -------------- | ---------------------- |
| baselink | lslidar        | -0.05  0  0.4  0  0  0 |
| baselink | camera         | 0.04  0  0.3  0  0  0  |
| baselink | imu            | 0.11  0  0.25  0  0  0 |





# 标定情况

## 相机zed2i标定
相机出厂内参：/usr/local/zed/settings/SN37510078.conf

使用kalibr
结果：~/hjd/biaoding/bag

第一次标定（标定板平整，粘贴在墙上，小车动，但是有一定运动模糊 对应包zed2i_imu）：~/hjd/biaoding/bag/zed2i_biaodingres

第二次标定（标定板不平整，手持，上下运动范围不够 对应包zed2i_laser）

第三次标定（标定板粘贴在纸板箱上，较为平整，已覆盖第二次包zed2i_laser）


## IMU标定

使用imu_utils
标定方法：https://blog.csdn.net/fang794735225/article/details/92804030

对应包/home/ouc/hjd/biaoding/bag/imu.bag
对应文件/home/hjd/biaoding/catkin_ws/src/imu_utils/data/myimu_data/my_imu_imu_param.yaml



## 相机雷达标定
相机更换位置后重新标定相机雷达的外参
github：https://github.com/MegviiRobot/CamLaserCalibraTool
使用~/hjd/biaoding/LaserCameraCal_ws/src/CamLaserCalibraTool 

结果：/home/ouc/hjd/biaoding/bag/lidar_zed2i外参

对应包~/hjd/biaoding/bag/zed2i_lidar_extrinsic_2026-11-06-10-50-46.bag

修改对应相机TF坐标系文件
/home/ouc/hjd/catkin_ws/src/zed-ros/zed-ros-wrapper/zed_wrapper/launch/zed2i.launch
