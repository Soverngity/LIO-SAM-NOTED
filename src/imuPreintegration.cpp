#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

/* 将最终优化过的里程计信息添加上后面imu里程计增加的里程计信息构成最新的imu里程计信息 */
class TransformFusion : public ParamServer
{
public:
    ofstream out_txt_raw_file;          // imu原始轨迹保存txt
    ofstream out_txt_preincremental_file;  // imu预积分轨迹保存txt
    std::mutex mtx;

    ros::Subscriber subImuRawOdometry;  // imu原始数据中的里程计数据
    ros::Subscriber subImuOdometry;     // 通过imu积分估计的雷达里程计信息订阅器
    ros::Subscriber subLaserOdometry;   // 最终优化后的里程计信息订阅器

    ros::Publisher pubImuOdometry;      // imu里程计信息发布器
    ros::Publisher pubImuPath;          // imu路径发布器

    ros::Publisher pubImuRawOdometry;   // imu原始里程计信息发布器
    ros::Publisher pubImuRawPath;       // imu原始路径发布器

    Eigen::Affine3f lidarOdomAffine;    //Transform<float,3,Affine> T = Translation3f(p) * AngleAxisf(a,axis) * Scaling(s); (4*4齐次矩阵变换+尺度变化-仿射)
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;    // tf数据接收
    tf::StampedTransform lidar2Baselink; // tf变换

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;    // IMU预积分里程计数据
    deque<nav_msgs::Odometry> imuRawOdomQueue; // SPAN-CPT里程计原始数据

    // TransformFusion这个类产生的数据没有被其它节点使用，只是单纯的为了rviz显示用，所以这个类可以去掉，不影响最后的建图结果
    TransformFusion()
    {
        // 如果雷达坐标系和base坐标系不一致，则获取雷达坐标系相对于base坐标系的转换关系
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink); //得到Lidar到base_link的transform
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }
        // IMU原始数据中的里程计数据订阅/navsat/odom
        subImuRawOdometry = nh.subscribe<nav_msgs::Odometry>("navsat/odom",   2000, &TransformFusion::imuRawOdometryHandler,   this, ros::TransportHints().tcpNoDelay());
        // 启动TCP_NODELAY，就意味着禁用了Nagle算法(数据只有在写缓存中累积到一定量之后，才会被发送出去，这样明显提高了网络利用率)，允许小包的发送。对于延时敏感型，同时数据传输量比较小的应用，开启TCP_NODELAY选项无疑是一个正确的选择
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        // IMU预计分话题接收:/odometry/imu_incremental
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());
        // 发布的话题
        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);   // 该话题没有被任何其它节点利用
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);

        pubImuRawOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic + "_raw", 2000);   // 该话题没有被任何其它节点利用
        pubImuRawPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu_raw/path", 1);
        if(savePATH) init();
    }
    void init()
    {
        time_t now_time = time(NULL);
        struct tm *p = gmtime(&now_time);
        char txtName[256] = {0};
        sprintf(txtName, "%s%d-%d-%d-%d:%02d:%02d", savePATHDirectory.c_str(), 1900+p->tm_year,1+p->tm_mon,p->tm_mday,8+p->tm_hour,p->tm_min,p->tm_sec);
        string txtNameStr = string(txtName);
        out_txt_raw_file.open(txtNameStr + "_raw.txt", ios::out | ios::trunc);
        out_txt_preincremental_file.open(txtNameStr + "_integral.txt", ios::out | ios::trunc);
    }

    // 里程计到仿射矩阵转换
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) // /lio_sam/mapping/odometry(lidar没有帧间匹配)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 调用工具类函数 获得从地图优化程序发布的里程计消息的仿射变换矩阵的形式
        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // odom和map坐标系重合, static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg); // 记录通过imu估计的雷达里程计信息(后面简称imu里程计信息)

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1) // 当没有订阅到最终优化后的里程计信息时，直接返回
            return;
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime) // 只取已优化过lidar_odom后的imu_odom
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack; // imu里程计增量
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre; // 优化后的里程计增量+后面imu里程计增量
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

        // 发布最新的里程计, 虽然发布但无节点接收, publish latest odometry
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // 由上边计算的里程计发布tf, publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // 0.1s发布一次imu path
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            if(savePATH)
            {
                out_txt_preincremental_file << setprecision(6)
                                            << pose_stamped.header.stamp << " "
                                            << pose_stamped.pose.position.x << " "
                                            << pose_stamped.pose.position.y << " "
                                            << pose_stamped.pose.position.z << " "
                                            << pose_stamped.pose.orientation.x << " "
                                            << pose_stamped.pose.orientation.y << " "
                                            << pose_stamped.pose.orientation.z << " "
                                            << pose_stamped.pose.orientation.w << endl;
            }
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0) // 只显示1s内的imu path
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }

    void imuRawOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {

        imuRawOdomQueue.push_back(*odomMsg); // 记录通过imu估计的雷达里程计信息(后面简称imu里程计信息)

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1) // 当没有订阅到最终优化后的里程计信息时，直接返回
            return;
        while (!imuRawOdomQueue.empty())
        {
            if (imuRawOdomQueue.front().header.stamp.toSec() <= lidarOdomTime) // 只取已优化过lidar_odom后的imu_odom
                imuRawOdomQueue.pop_front();
            else
                break;
        }

        // 发布最新的imu原始里程计
        nav_msgs::Odometry laserOdometry = imuRawOdomQueue.back();
        // odom从imu坐标系转换到雷达坐标系
        laserOdometry = odomConverter(laserOdometry);
        pubImuRawOdometry.publish(laserOdometry);

        // 由上边计算的里程计发布tf, publish tf
        static tf::TransformBroadcaster tfOdom2ImuLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_imulink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, "imu_link");
        tfOdom2ImuLink.sendTransform(odom_2_imulink);

        // publish IMU path
        static nav_msgs::Path imuRawPath;
        static double last_raw_path_time = -1;
        double imuTime = imuRawOdomQueue.back().header.stamp.toSec();
        // 0.1s发布一次imu path
        if (imuTime - last_raw_path_time > 0.1)
        {
            last_raw_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;

            pose_stamped.header.stamp = imuRawOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuRawPath.poses.push_back(pose_stamped);
            if(savePATH)
            {
                out_txt_raw_file << setprecision(6)
                                 << pose_stamped.header.stamp << " "
                                 << pose_stamped.pose.position.x << " "
                                 << pose_stamped.pose.position.y << " "
                                 << pose_stamped.pose.position.z << " "
                                 << pose_stamped.pose.orientation.x << " "
                                 << pose_stamped.pose.orientation.y << " "
                                 << pose_stamped.pose.orientation.z << " "
                                 << pose_stamped.pose.orientation.w << endl;
            }
//            while(!imuRawPath.poses.empty() && imuRawPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0) // 只显示1s内的imu path
//                imuRawPath.poses.erase(imuRawPath.poses.begin());
            if (pubImuRawPath.getNumSubscribers() != 0)
            {
                imuRawPath.header.stamp = imuRawOdomQueue.back().header.stamp;
                imuRawPath.header.frame_id = odometryFrame;
                pubImuRawPath.publish(imuRawPath);
            }
        }
    }
    ~TransformFusion()
    {
        out_txt_raw_file.close();
        out_txt_preincremental_file.close();
    }
};


class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    ros::Subscriber subImu;           // imu信息订阅器
    ros::Subscriber subOdometry;      // 最终优化后的里程计增量信息(用来矫正imu的偏置)
    ros::Publisher pubImuOdometry;    // 估计的imu里程计信息发布器(其实是通过imu估计的雷达里程计信息)

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;   // 先验位置噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;    // 先验速度噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;   // 先验偏置噪声
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;                    // 上一时刻估计imu的位姿信息
    gtsam::Vector3 prevVel_;                   // 上一时刻估计imu的速度信息
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    // 从四元数旋转和平移矩阵构建gtsam::Pose3
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);
        // 定义进行imu积分的imu传感器信息
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity); //向上的z轴的重力默认参数,牧马人此值为0(因为惯导输出的值已经不含重力值)
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // 加速度计的白噪声 acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // 陀螺仪的白噪声gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // 通过速度积分位置信息引入的噪声 error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // 初始化imu偏置信息assume zero initial bias
        // 初始化位姿、速度、偏置的噪声
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        // 根据上面的参数，定义两个imu预积分器，一个用于imu信息处理线程，一个用于优化线程
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
    }

    void resetOptimization()
    {
        // 重置isam2优化器
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        // 重置初始化非线性因子图
        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;       // 标识执行过第一次优化（用在imuHandler）
        systemInitialized = false;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;

        // 转换消息数据为gtsam的3d位姿信息
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false; // 协方差标识退化
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system 矫正过程的初始化
        if (systemInitialized == false)
        {
            resetOptimization();

            // 丢弃老的imu信息 pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // 通过最终优化过的雷达位姿初始化先验的位姿信息并添加到因子图中, initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // 初始化先验速度信息为0并添加到因子图中, initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // 初始化先验偏置信息为0并添加到因子图中, initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // 设置变量的初始估计值, add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // 将因子图更新到isam2优化器中, optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();
            // 重设imu预积分器，一个用于imu信息处理线程，一个用于优化线程
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            key = 1;
            systemInitialized = true;
            return;
        }


        // 当isam2规模太大时, 进行边缘化, 重置优化器和因子图, reset graph for speed
        if (key == 100)
        {
            // 获取最新关键帧的协方差, get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // 重置isam2优化器和因子图, reset graph
            resetOptimization();
            // 按最新关键帧的协方差将位姿、速度、偏置因子添加到因子图中, add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // 并用最新关键帧的位姿、速度、偏置初始化对应的因子, add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // 并将最新初始化的因子图更新到重置的isam2优化器中, optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. 预积分imu数据并进行优化 integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // 对相邻两次优化之间的imu帧进行积分, 并移除, pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                //// 初始时, lastImuT_opt=-1, dt取1/500=0.002, 是否应该根据自己的IMU数据频率改为1/100？？？
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);

                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // 将imu因子添加到因子图中, add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        // 该因子涉及的优化变量包括:上一帧和当前帧的位姿和速度，上一帧的偏置，相邻两关键帧之间的预积分结果
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        //  将imu偏置因子添加到因子图中, add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // 添加当前关键帧位姿因子, add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // 设置当前关键帧位姿因子、速度因子和偏置因子的【初始值】, insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // 将最新关键帧相关的因子图更新到isam2优化器中，并进行优化, optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // 获取当前关键帧的优化结果, 并将结果置为先前值, Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // 利用优化后的imu偏置信息重置imu预积分对象, Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // 对优化结果进行失败检测:当速度和偏置太大时, 则认为优化失败, check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        // 当速度太大，则认为失败
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        // 当偏置太大，则认为失败
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 利用变换参数将数据从imu坐标系转到了雷达坐标系, 其实也就是获得雷达运动的加速度、角速度和姿态信息
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);
        // 加入队列, 分别为优化前后的imu数据
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);
        // 检查是否执行过第一次优化（需要先在odomHandler优化后才能继续后续操作）
        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        //// 初始时, lastImuT_imu=-1, dt取1/500=0.002, 是否应该根据自己的IMU数据频率改为1/100？？？
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // 对单imu消息使用gtsam进行预积分, integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // 利用上一时刻的imu里程计状态信息PVQ和偏置信息, 预积分当前时刻imu里程计状态信息PVQ, predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // 前边的imu转lidar只是做了旋转, 在此做再做平移变换, transform imu pose to lidar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);  // 获得估计的雷达位姿信息
//        cout << "imuPose: " << " x: " << imuPose.translation().x() << " y: " << imuPose.translation().y() << " z: " << imuPose.translation().z() << endl;
//        cout << "imuRotation: " << " ox: " << imuPose.rotation().toQuaternion().x() << " oy: " << imuPose.rotation().toQuaternion().y() << " oz: " << imuPose.rotation().toQuaternion().z() << " ow:" << imuPose.rotation().toQuaternion().w() << endl;
//        cout << "lidarPose: " << " x: " << lidarPose.translation().x() << " y: " << lidarPose.translation().y() << " z: " << lidarPose.translation().z() << endl;
//        cout << "lidarRotation: " << " ox: " << lidarPose.rotation().toQuaternion().x() << " oy: " << lidarPose.rotation().toQuaternion().y() << " oz: " << lidarPose.rotation().toQuaternion().z() << " ow: " << lidarPose.rotation().toQuaternion().w() << endl;

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        // 角速度直接从imu消息读取 + 偏置
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");

    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

    ros::MultiThreadedSpinner spinner(5);
    spinner.spin();

    return 0;
}