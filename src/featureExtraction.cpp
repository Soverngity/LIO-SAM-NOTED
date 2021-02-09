#include "utility.h"
#include "lio_sam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;              // 雷达点云信息订阅器

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;                 // 角点特征发布器
    ros::Publisher pubSurfacePoints;                // 平面点特征发布器

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;    // 角点点云
    pcl::PointCloud<PointType>::Ptr surfaceCloud;   // 平面点点云

    pcl::VoxelGrid<PointType> downSizeFilter;       // 降采样滤波器(降低角点和平面点密度)

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;      // 点云顺滑性缓存器(每个元素包含点的曲率和索引)
    float *cloudCurvature;                          // 点云中点的曲率
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        
        initializationValue();
    }

    void initializationValue()
    {
        //cxp 20210127新增,对类中指针成员变量的初始化
        cloudCurvature = nullptr;
        cloudNeighborPicked = nullptr;
        cloudLabel = nullptr;

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction
        // 计算点云中点的曲率
        calculateSmoothness();
        // 标记遮挡点和平行光束点，避免后面进行错误的特征提取
        markOccludedPoints();
        // 特征提取(平面点和角点)
        extractFeatures();
        // 发布特征点信息
        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        // 这里跟loam源码基本很像,从第五个点开始到结束前5个,计算某个点的平滑度
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 计算当前点与左右相邻的10个点之间的深度差的总和
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];
            // 并将该深度差的总和的平方作为当前点的曲率信息
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 先将是否被选择以及标签值置0
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // 缓存点的曲率信息，便于后面依据曲率对点进行排序 cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }
    // 标记遮挡点和平行光束点，参考LOAM论文第4幅图
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // 标记遮挡点和平行光束点 mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // 两个点的深度也就是雷达到障碍物的距离 occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            // 获取相邻两点之间的列索引差
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));
            // 如果列索引差较小，即两点在扫描角度上靠的很近
            if (columnDiff < 10){
                // 如果深度值差的较大，则将相邻的5个点认为是遮挡点，并标记为已选择过，后面不会再对这些点进行特征提取10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));
            // 如果相邻深度差都较大，则认为当前点为平行光束点(即激光近似水平射向反射平面)，并标记为已选择过，后面不会对这些点进行特征提取
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
            // 将每根线均分成6段，然后分别对每一段进行特征提取
            for (int j = 0; j < 6; j++)
            {
                // 计算每段的起始和结束点索引
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;
                // 对每段点云数据依据曲率进行由小到大排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                // 由于sp->ep的点云已经按曲率有小到大排序过，此处从ep开始检索，意味着从曲率最大点开始检索
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;   // 读取当前检索点对应的索引
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        // 当点的曲率超过设定的阈值，则认为是角点，并缓存
                        largestPickedNum++;
                        if (largestPickedNum <= 20){    // 每段最多提取20个角点
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        // 对当前检索点的左右各5个相邻点进行列索引判断，如果靠的比较近，则将这些相邻点置为选择过的，这样就可以保证不在这些点处提取角点，避免了角点分布过于密集
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 由于sp->ep的点云已经按曲率有小到大排序过，此处从sp开始检索，意味着从曲率最小点开始检索
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;
                        // 对当前检索点的左右各5个相邻点进行列索引判断，如果靠的比较近，则将这些相邻点置为选择过的，这样就可以保证不在这些点处提取平面点，避免了平面点分布过于密集
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 根据标记获取平面点
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
            // 对平面点进行降采,体素滤波就是对提取出的平面点云进行下采样
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);
            // 将每条线提取且下采样后的平面点加到最终的平面点云上
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}