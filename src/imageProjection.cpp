#include "utility.h"
#include "lio_sam/cloud_info.h"

//注册VelodynePointXYZIRT新的点云类型
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D                  //添加pcl里xyz+padding
    PCL_ADD_INTENSITY;               //添加pcl里intensity
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 确保定义新类型点云内存与SSE对齐
} EIGEN_ALIGN16;                    // 强制SSE填充以正确对齐内存
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

//注册OusterPointXYZIRT新的点云类型
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

// 继承utility.h的ParamServer类,获得nh以及各话题名等变量
class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud; // 雷达消息订阅器,未使用
    ros::Publisher  pubLaserCloud; // 未使用
    
    ros::Publisher pubExtractedCloud; // 发布雷达点云,cloud_deskewed话题
    ros::Publisher pubLaserCloudInfo; // 发布雷达点云信息,cloud_info话题

    ros::Subscriber subImu;                    // imu消息订阅器
    std::deque<sensor_msgs::Imu> imuQueue;     // imu消息缓存队列

    ros::Subscriber subOdom;                   // imu里程计增量订阅器
    std::deque<nav_msgs::Odometry> odomQueue;  // imu里程计缓存队列

    std::deque<sensor_msgs::PointCloud2> cloudQueue; // 原始雷达消息缓存队列
    sensor_msgs::PointCloud2 currentCloudMsg;        // 队列头部雷达消息,一帧

    // 进行点云偏斜矫正时所需的通过imu积分获得的imu姿态信息
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse; // typedef Transform<float,3,Affine> Affine3f; Affine为0x2,是一种仿射变换

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn; // 输入点云,using PointXYZIRT = VelodynePointXYZIRT;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;

    // extractedCloud为从fullCloud得到的有效点云，随后发布给Rviz，并赋值给cloudInfo
    pcl::PointCloud<PointType>::Ptr   fullCloud;      // typedef pcl::PointXYZI PointType; 完整点云
    pcl::PointCloud<PointType>::Ptr   extractedCloud; // 偏斜矫正后点云

    int deskewFlag; // 点云去倾斜可用标志,初始化为0,1为可用,-1为不可用
    cv::Mat rangeMat;  // 点云投影获得的深度图

    // 进行点云偏斜矫正时所需的通过imu里程计获得的imu位置增量
    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur; // 雷达帧扫描开始时间戳
    double timeScanEnd; // 雷达帧扫描结束时间戳
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 启动TCP_NODELAY，就意味着禁用了Nagle算法(数据只有在写缓存中累积到一定量之后，才会被发送出去，这样明显提高了网络利用率)，允许小包的发送。对于延时敏感型，同时数据传输量比较小的应用，开启TCP_NODELAY选项无疑是一个正确的选择
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay()); //incremental->增量
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布的话题
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters(); // 多此一举？？？

        //用于设置控制台输出的信息等级
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    // 因为变量仅声明，定义变量分配存储空间
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN); //N_SCAN*Horizon_SCAN=16*1800

        //assign分配给vector数组N_SCAN个0，即将startRingIndex、endRingIndex数组初始化为N_SCAN个0
        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);
        // 与上同理
        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear(); //points.clear (); width = 0; height = 0;
        extractedCloud->clear();
        // 重置距离图像投影的距离矩阵,FLT_MAX:float能表示的最大值
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        // imu通过积分获得的姿态信息
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        //imuConverter在utility.h中定义,将imu坐标系的数据转到雷达坐标系下
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
//         cout << std::setprecision(6);
//         cout << "IMU acc: " << endl;
//         cout << "x: " << thisImu.linear_acceleration.x <<
//               ", y: " << thisImu.linear_acceleration.y <<
//               ", z: " << thisImu.linear_acceleration.z << endl;
//         cout << "IMU gyro: " << endl;
//         cout << "x: " << thisImu.angular_velocity.x <<
//               ", y: " << thisImu.angular_velocity.y <<
//               ", z: " << thisImu.angular_velocity.z << endl;
//         double imuRoll, imuPitch, imuYaw;
//         tf::Quaternion orientation;
//         tf::quaternionMsgToTF(thisImu.orientation, orientation);
//         tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
//         cout << "IMU roll pitch yaw: " << endl;
//         cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 将点云信息加入缓存队列,并将队列头部点云从ROS格式转换为PCL格式赋给currentCloudMsg
        if (!cachePointCloud(laserCloudMsg))
            return;
        // 点云偏斜矫正所需的imu数据的预处理(计算雷达帧扫描开始和结束时间戳之间的imu相对位姿变换)
        if (!deskewInfo())
            return;

        //// 上面一部分结束了用于去畸变的相关参数的获取,但是没有进行去畸变的工作,具体的工作在后面进行

        // 对点云进行偏斜矫正，并投影到深度图中
        projectPointCloud();

        // 点云提取，确定每根线的起始和结束点索引，并提取出偏斜矫正后点云及对应的点云信息
        cloudExtraction();

        // 点云发布
        publishClouds();

        // 对每一帧点云进行参数重置
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 点云信息存入队列,cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        // 点云太少就退出,其实似乎等于2也能用了,不过通常来说不会出现这样太少的情况所以不用纠结,因为用的时候还都是帧间的关系
        if (cloudQueue.size() <= 2)
            return false;

        // 点云队列先进先出,存到currentCloudMsg其类型是sensor_msgs::PointCloud  convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {
            // 将ROS雷达点云消息转换成pcl点云数据形式
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // 获取当前帧的时间戳
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        //// 计算当前帧扫描结束时时间戳，为什么就能获得？？？点云中的点按时间排序？？？time是个相对于当前帧开始的相对时间
        //// 查看Pandar40的驱动程序，未对点云进行排序，需进行调试查看！！！但是Pandar40点云header的时间戳赋值为points[0].time是否是相同的道理？
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;
        // printf("timeScanCur:%lf, timeScanEnd:%lf", timeScanCur, timeScanEnd);

        // 判断点云是否不包含nan或者inf值,若包含is_dense就为false,check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel,只检查一次
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            // 没有ring字段直接报错,KITTI数据集转成bag包时,添加了此字段
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            // KITTI数据集就没有此字段,因此此处只是warning,KITTI已经进行了点云去畸变
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        // std::lock_guard函数结束自动释放锁
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        // 通过积分获取当前雷达帧扫描开始和结束时间戳内的imu姿态信息(主要为获取imu帧相对姿态信息)
        imuDeskewInfo();
        // 获得imu里程计在当前雷达帧扫描开始和结束时间戳内的起始和结束帧，并计算两者之间的位姿变换(主要为获取imu帧相对位置信息)
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        //这个参数在地图优化mapOptmization.cpp程序中用到,首先为false,完成相关操作后置true
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            // 丢弃早于当前雷达帧开始扫描时间戳的缓存的imu帧,设置0.01阈值（100Hz IMU阈值是否需要更改？？？）
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            // 通过header得到IMU时间戳
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // 用四元数表示的imu姿态信息转换成欧拉角表示, get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur) //在此再次做时间判断
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 初始化第一帧imu姿态信息为0
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // 获取imu角速度信息 get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // 积分imu的姿态信息,用于后续的去畸变 integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;
        // 这里将该变量置true 在地图优化程序中会用到
        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        // 这个参数在地图优化mapOptmization.cpp程序中用到,首先为false,完成相关操作后置true
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // 获得每一帧扫描起始时刻的里程计消息,get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];
            // 自定义函数,return msg->header.stamp.toSec();
            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // 将imu里程计帧的姿态信息转换为欧拉角表示
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // 将imu里程计的位姿记录，并将被发布出去用于地图优化的初始值 Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;
        // 获得一帧扫描末尾的里程计消息,这个就跟初始位姿估计没有关系,只是用于去畸变,运动补偿
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        // 位姿协方差矩阵判断？？？round()为四舍五入函数
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 获得imu里程计起始帧位姿
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 获得imu里程计结束帧位姿
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 获得起始和结束帧之间的相对转换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 获得起始和结束帧之间的位姿增量，其中姿态用欧拉角形式表示
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        // 查找第一个时间戳大于等于当前点的imu数据指针
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // 如果不存在时间戳上大于当前点的imu数据帧，则直接返回最近的imu数据帧姿态信息
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 如果存在时间戳大于当前点的imu数据帧，则利用该帧前一帧和该帧进行插值，获取当前点时间戳对应的姿态信息，类似积分
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

         if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
             return;

         float ratio = relTime / (timeScanEnd - timeScanCur);

         *posXCur = ratio * odomIncreX;
         *posYCur = ratio * odomIncreY;
         *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        // deskewFlag标志可用也即有time字段,并且imuDeskewInfo函数证明imu旋转角信息可用
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        // 当前点采集时的时间戳
        double pointTime = timeScanCur + relTime;

        // 调用本程序文件内函数,获得当前点时间戳对应的imu姿态变化
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        // 调用本程序文件内函数,获得当前点时间戳对应imu位置变换(由于移动较慢，此处直接置为0)，低速点云去畸变主要是旋转
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        // 获取第一个点时间戳对应的位姿变化，起始变换矩阵赋初值再取逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // 把点投影到每一帧扫描的起始时刻，参考Loam里的方案 transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 变换当前点到第一个点所在的坐标系也即雷达坐标系(至此完成偏斜矫正)，得到去畸变后的点
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // 点云数据按线束,按行列保存.如果运行算法不报错,但是RVIZ中没有任何显示 重点排查该函数 range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 得到点的距离,距离是m
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange) //min为1m,max为1000m
                continue;
            // 得到线束号
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 针对每个点进行降采样,而不是针对线
            if (rowIdn % downsampleRate != 0)
                continue;
            // 计算激光点的水平角度
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            // 水平角分辨率 360/1800
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            // 计算在距离图像上点属于哪一列
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            // 超了一圈的情况
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 初始化为全FLT_MAX,如果已经投影过就continue
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            //对当前【点】进行去畸变,需要雷达的time的field
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            // fullCloud填充内容 这里是运动补偿去畸变后的点云
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // 确定每根线的起始和结束点索引，并提取出去畸变后的点云，extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 每根线上点云信息的起始索引未从起始点开始，原因是该部分提取出来的点云是用于特征提取的，起始点附近的点都无法有效计算曲率
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // publishCloud将pcl类型转换为ROS消息格式并发布提取出的有效点云
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        // cloudInfo本身即为ROS消息格式，故不需要转换，发布激光点云信息(包括每根线的起始和结束点索引、点深度、列索引)
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    // 多线程接受订阅话题，避免订阅多来源话题一个回调函数导致的阻塞
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
