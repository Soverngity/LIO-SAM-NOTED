# Cloud Info
Header header 

# 在imageProjectin.cpp中的cloudExtraction函数进行赋值，用于后续特征提取
# 点云中的点被投影到深度图中，并按行优先的方式用数组来存储这个深度图，下面这两个数据分别是每根线(一般是16线)起始激光点对应的有效深度图（extractedCloud）起始和结束数组索引
int32[] startRingIndex
int32[] endRingIndex
int32[]  pointColInd # point column index in range image # 每个激光点在完整深度图中(fullCloud)的列索引
float32[] pointRange # point range # 保存深度图的数组

# imu和imu里程计是否可用于匹配的初始位姿估计(imuDeskewInfo()有没有成功执行)
int64 imuAvailable
int64 odomAvailable

# 当imu是可得到时，提供的初始姿态估计 Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# 当imu里程计是可得到时，提供的初始位姿估计 Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# # 点云相关信息 Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature