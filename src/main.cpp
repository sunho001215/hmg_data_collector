#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include <opencv2/core/eigen.hpp>
#include "ros/ros.h"
#include "ros/package.h"

#include <sensor_msgs/PointCloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
 #include <message_filters/sync_policies/approximate_time.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Transform.h>
#include <tf/LinearMath/Vector3.h>
#include "tf_conversions/tf_eigen.h"
#include "geometry_msgs/Point32.h"

#include <nav_msgs/Odometry.h>

#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"

#include "hellocm_msgs/ObjectSensor.h"
#include "hellocm_msgs/ObjectSensorObj.h"
#include "hellocm_msgs/ObservPoint.h"

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>


using namespace message_filters;
using namespace sensor_msgs;
using namespace std; 
using namespace std::chrono; 
using namespace cv;
using namespace nav_msgs;
using namespace Eigen;
using namespace hellocm_msgs;

const double cm_per_pixel = 5.0;
const double map_scale = 100.0/cm_per_pixel;
const int map_height = 800;
const int map_width = 800;
const int img_size = 800;
const double pi = M_PI;
const int img_cor_x = map_width/2;
const int img_cor_y = map_scale * 10;

string get_string(int n, int d){
    if(n>pow(10,d)) {
        ROS_ERROR("N is too large... can't change into string");
        return "";
    }
    if(n==0) return "0000";
    int digit = int(log10(n));
    string rt = "";
    for(int i=0;i<d-1-digit;i++) rt += "0";
    rt += to_string(n);
    return rt;
}

geometry_msgs::Point32 vector_to_point(tf::Vector3 v){
    geometry_msgs::Point32 p;
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
    return p;
}

class DataCollector
{
    private:
        ros::NodeHandle nh;
        message_filters::Subscriber<sensor_msgs::PointCloud> pc_sub1;
        message_filters::Subscriber<sensor_msgs::PointCloud> pc_sub2;
        message_filters::Subscriber<nav_msgs::Odometry> odo_sub;
        message_filters::Subscriber<hellocm_msgs::ObjectSensor> obj_sub;
        message_filters::Subscriber<sensor_msgs::Image> img_1;
        message_filters::Subscriber<sensor_msgs::Image> img_2;
        message_filters::Subscriber<sensor_msgs::Image> img_3;
        typedef sync_policies::ApproximateTime<PointCloud, PointCloud, Odometry, ObjectSensor, Image, Image, Image> SyncPolicy;
        typedef Synchronizer<SyncPolicy> Sync;
        boost::shared_ptr<Sync> sync;
        tf::Transform lidar_tf_1;
        tf::Transform lidar_tf_2;      
        tf::Quaternion id;
        tf::Transform obj_sensor_tf; // object sensor tf for car
        tf::Transform odom_tf;
        tf::Vector3 vehicle_vel;
        tf::Vector3 vehicle_ang;
        tf::Transform cam_tf_1, cam_tf_2, cam_tf_3;
        tf::Transform cam_tf;
        cv::Mat homography_1, homography_2, homography_3;
        cv::Mat front_mask, left_mask, right_mask;
        int data_num;
        bool mask_gen;
        
    public:
        DataCollector(){
            pc_sub1.subscribe(nh, "/pointcloud/vlp", 10);
            pc_sub2.subscribe(nh, "/pointcloud/os1", 10);
            odo_sub.subscribe(nh, "/Odometry", 10);
            obj_sub.subscribe(nh, "/Object_Sensor_front", 10);
            img_1.subscribe(nh, "/color1/image_raw", 10);
            img_2.subscribe(nh, "/color2/image_raw", 10);
            img_3.subscribe(nh, "/color3/image_raw", 10);

            sync.reset(new Sync(SyncPolicy(10), pc_sub1, pc_sub2, odo_sub, obj_sub, img_1, img_2, img_3));
            sync->registerCallback(boost::bind(&DataCollector::callback, this, _1, _2, _3, _4, _5, _6, _7));

            // Lidar
            tf::Quaternion q_1;
            tf::Vector3 v_1 = tf::Vector3(2.0, 0.0, 2.0);
            q_1.setRPY(0,0,90.0*M_PI/180.0);
            lidar_tf_1 = tf::Transform(q_1,v_1);

            tf::Quaternion q_2;
            tf::Vector3 v_2 = tf::Vector3(2.5, 0.0, 2.0);
            q_2.setRPY(0,0,0);
            lidar_tf_2 = tf::Transform(q_2,v_2);

            id.setRPY(0,0,0);

            // Object sensor
            tf::Quaternion obj_q;
            tf::Vector3 obj_v = tf::Vector3(-15.0, 0.0, 0.0);
            obj_q.setRPY(0, 0, 0);
            obj_sensor_tf = tf::Transform(obj_q, obj_v);

            // Camera coordinate tf
            tf::Quaternion cam_q;
            tf::Vector3 cam_v = tf::Vector3(0, 0, 0);
            cam_q.setRPY(-90.0*M_PI/180.0, 0, -90.0*M_PI/180.0);
            cam_tf = tf::Transform(cam_q, cam_v);
            

            Matrix<double, 3, 4> P;
            P(0,0) = 346.43486958; P(0,1) = 0.0; P(0,2) = 600.0; P(0,3) = 0.0;
            P(1,0) = 0.0; P(1,1) = 346.43486958; P(1,2) = 400.0; P(1,3) = 0.0;
            P(2,0) = 0.0; P(2,1) = 0.0; P(2,2) = 1.0; P(2,3) = 0.0;

            Matrix<double, 3, 3> dist_to_img;
            dist_to_img.setZero();
            dist_to_img(0,1) = -map_scale; dist_to_img(0,2) = 20 * map_scale;
            dist_to_img(1,0) = -map_scale; dist_to_img(1,2) = 30 * map_scale;
            dist_to_img(2,2) = 1;

            // Camera 1
            tf::Quaternion cam_q_1;
            tf::Transform temp_tf_1;
            Eigen::Affine3d cam_affine_1;
            Matrix<double, 3, 4> mat_3x4_1;
            Matrix<double, 3, 3> vehicle_to_cam_1;
            Matrix<double, 3, 3> homo_1;
            tf::Vector3 cam_v_1 = tf::Vector3(2.1, -0.7, 2.6);
            cam_q_1.setRPY(0, 25.0*M_PI/180.0, -100*M_PI/180.0);
            temp_tf_1 = tf::Transform(cam_q_1, cam_v_1);
            cam_tf_1 = temp_tf_1 * cam_tf;
            tf::transformTFToEigen(cam_tf_1.inverse(), cam_affine_1);
            mat_3x4_1 = P * cam_affine_1.matrix();
            vehicle_to_cam_1(0,0) = mat_3x4_1(0,0); vehicle_to_cam_1(0,1) = mat_3x4_1(0,1); vehicle_to_cam_1(0,2) = mat_3x4_1(0,3);
            vehicle_to_cam_1(1,0) = mat_3x4_1(1,0); vehicle_to_cam_1(1,1) = mat_3x4_1(1,1); vehicle_to_cam_1(1,2) = mat_3x4_1(1,3);
            vehicle_to_cam_1(2,0) = mat_3x4_1(2,0); vehicle_to_cam_1(2,1) = mat_3x4_1(2,1); vehicle_to_cam_1(2,2) = mat_3x4_1(2,3);
            homo_1 = dist_to_img * vehicle_to_cam_1.inverse();
            cv::eigen2cv(homo_1,homography_1);

            // Camera 2
            tf::Quaternion cam_q_2;
            tf::Transform temp_tf_2;
            Eigen::Affine3d cam_affine_2;
            Matrix<double, 3, 4> mat_3x4_2;
            Matrix<double, 3, 3> vehicle_to_cam_2;
            Matrix<double, 3, 3> homo_2;
            tf::Vector3 cam_v_2 = tf::Vector3(2.3, 0.0, 2.6);
            cam_q_2.setRPY(0, -15.0*M_PI/180.0, 0);
            temp_tf_2 = tf::Transform(cam_q_2, cam_v_2);
            cam_tf_2 = temp_tf_2 * cam_tf;
            tf::transformTFToEigen(cam_tf_2.inverse(), cam_affine_2);
            mat_3x4_2 = P * cam_affine_2.matrix();
            vehicle_to_cam_2(0,0) = mat_3x4_2(0,0); vehicle_to_cam_2(0,1) = mat_3x4_2(0,1); vehicle_to_cam_2(0,2) = mat_3x4_2(0,3);
            vehicle_to_cam_2(1,0) = mat_3x4_2(1,0); vehicle_to_cam_2(1,1) = mat_3x4_2(1,1); vehicle_to_cam_2(1,2) = mat_3x4_2(1,3);
            vehicle_to_cam_2(2,0) = mat_3x4_2(2,0); vehicle_to_cam_2(2,1) = mat_3x4_2(2,1); vehicle_to_cam_2(2,2) = mat_3x4_2(2,3);
            homo_2 = dist_to_img * vehicle_to_cam_2.inverse();
            cv::eigen2cv(homo_2,homography_2);

            // Camera 3
            tf::Quaternion cam_q_3;
            tf::Transform temp_tf_3;
            Eigen::Affine3d cam_affine_3;
            Matrix<double, 3, 4> mat_3x4_3;
            Matrix<double, 3, 3> vehicle_to_cam_3;
            Matrix<double, 3, 3> homo_3;
            tf::Vector3 cam_v_3 = tf::Vector3(2.1, 0.7, 2.6);
            cam_q_3.setRPY(0, 25.0*M_PI/180.0, 100*M_PI/180.0);
            temp_tf_3 = tf::Transform(cam_q_3, cam_v_3);
            cam_tf_3 = temp_tf_3 * cam_tf;
            tf::transformTFToEigen(cam_tf_3.inverse(), cam_affine_3);
            mat_3x4_3 = P * cam_affine_3.matrix();
            vehicle_to_cam_3(0,0) = mat_3x4_3(0,0); vehicle_to_cam_3(0,1) = mat_3x4_3(0,1); vehicle_to_cam_3(0,2) = mat_3x4_3(0,3);
            vehicle_to_cam_3(1,0) = mat_3x4_3(1,0); vehicle_to_cam_3(1,1) = mat_3x4_3(1,1); vehicle_to_cam_3(1,2) = mat_3x4_3(1,3);
            vehicle_to_cam_3(2,0) = mat_3x4_3(2,0); vehicle_to_cam_3(2,1) = mat_3x4_3(2,1); vehicle_to_cam_3(2,2) = mat_3x4_3(2,3);
            homo_3 = dist_to_img * vehicle_to_cam_3.inverse();
            cv::eigen2cv(homo_3,homography_3);

            // mask
            front_mask = cv::Mat(map_height, map_width,CV_8UC1);
            right_mask = cv::Mat(map_height, map_width,CV_8UC1);
            left_mask = cv::Mat(map_height, map_width,CV_8UC1);
            
            for(int i=0; i<map_height; i++)
            {
                for(int j=0; j<map_width; j++)
                {
                    if(i > map_width/2){
                        right_mask.at<uchar>(j,i) = 255;
                        left_mask.at<uchar>(j,i) = 0;
                    }
                    else{
                        left_mask.at<uchar>(j,i) = 255;
                        right_mask.at<uchar>(j,i) = 0;
                    }
                    if(j < map_height/2 + 150){
                        front_mask.at<uchar>(j,i) = 255;
                    }
                    else{
                        front_mask.at<uchar>(j,i) = 0;
                    }
                }
            }

            mask_gen = false;
            data_num = 1;
        }

        void callback(const PointCloud::ConstPtr& pc_msg1, const PointCloud::ConstPtr& pc_msg2 , const Odometry::ConstPtr& odo_msg, const ObjectSensor::ConstPtr& obj_msg, const Image::ConstPtr& img_msg1, const Image::ConstPtr& img_msg2, const Image::ConstPtr& img_msg3);
};

void DataCollector::callback(const PointCloud::ConstPtr& pc_msg1, const PointCloud::ConstPtr& pc_msg2 , const Odometry::ConstPtr& odo_msg, const ObjectSensor::ConstPtr& obj_msg, const Image::ConstPtr& img_msg1, const Image::ConstPtr& img_msg2, const Image::ConstPtr& img_msg3)
{
    auto start = std::chrono::high_resolution_clock::now();

    Mat Lidar_BEV_map(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    Mat Image_BEV_map(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    Mat Camera_BEV_left(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    Mat Camera_BEV_front(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    Mat Camera_BEV_right(map_height, map_width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    double z_min = 98765432.0;

    //=================== Label Data ===================//
    geometry_msgs::Quaternion q = odo_msg->pose.pose.orientation;
    geometry_msgs::Point p = odo_msg->pose.pose.position;
    odom_tf = tf::Transform(tf::Quaternion(q.x,q.y,q.z,q.w),tf::Vector3(p.x,p.y,p.z));
    vehicle_vel = tf::Vector3(odo_msg->twist.twist.linear.x, odo_msg->twist.twist.linear.y, odo_msg->twist.twist.linear.z);
    vehicle_ang = tf::Vector3(odo_msg->twist.twist.angular.x, odo_msg->twist.twist.angular.y, odo_msg->twist.twist.angular.z);

    vector<vector<double>> output_data;

    for(hellocm_msgs::ObjectSensorObj object : obj_msg->Objects){
        tf::Vector3 ref_position = tf::Vector3(object.RefPnt.ds[0],object.RefPnt.ds[1],object.RefPnt.ds[2]);
        tf::Quaternion ref_orientation;
        ref_orientation.setRPY(object.RefPnt.r_zyx[0],object.RefPnt.r_zyx[1],object.RefPnt.r_zyx[2]);
        tf::Transform ref_T = obj_sensor_tf * tf::Transform(ref_orientation, ref_position);
        tf::Vector3 obj_position = tf::Vector3(object.l/2,0,0);
        tf::Transform obj_T = ref_T * tf::Transform(id, obj_position);
        tf::Vector3 position = obj_T.getOrigin();
        
        double x, y, rad;
        x = static_cast<double>(((map_width-1)-position.y() * map_scale - img_cor_x)/img_size);
        y = static_cast<double>(((map_height-1)-position.x() * map_scale - img_cor_y)/img_size);
        rad = object.RefPnt.r_zyx[2] + M_PI/2.0;

        if(rad < -M_PI){
            rad = rad + 2*M_PI;
        }
        else if(rad > M_PI){
            rad = rad - 2*M_PI;
        }

        if(x>=0 && x<=1 && y>=0 && y<=1){
            vector<double> temp;
            temp.push_back(object.Name[0]);
            temp.push_back(x);
            temp.push_back(y);
            temp.push_back(object.l*map_scale/img_size);
            temp.push_back(object.w*map_scale/img_size);
            temp.push_back(rad);
            output_data.push_back(temp);
        }
    }

    if(output_data.size() == 0){
        return;
    }
    //=================== End ===================//

    //=================== Lidar Data ===================//
    if(pc_msg1->points.size() == 0 || pc_msg2->points.size() == 0){
        return;
    }

    for(geometry_msgs::Point32 point : pc_msg1->points){
        geometry_msgs::Point32 new_point;
        tf::Vector3 p = tf::Vector3(point.x, point.y, point.z);
        tf::Transform m = lidar_tf_1 * tf::Transform(id, p);
        tf::Vector3 v = m.getOrigin();
        new_point = vector_to_point(v);
        if (!((new_point.x >= 0) && (new_point.x <= 4.5) && (new_point.y >= -1) && (new_point.y <= 1)))
        {
            if ((new_point.x * map_scale + img_cor_y >= 0) && (new_point.x * map_scale + img_cor_y < map_height)){
                if((new_point.y * map_scale + img_cor_x >= 0) && (new_point.y * map_scale + img_cor_x < map_width)){
                    int y = (map_height - 1) - static_cast<int>(new_point.x * map_scale + img_cor_y);
                    int x = (map_width - 1) - static_cast<int>(new_point.y * map_scale + img_cor_x);
                    if (Lidar_BEV_map.at<cv::Vec3f>(y,x)[2] == 0.0){
                        Lidar_BEV_map.at<cv::Vec3f>(y,x)[2] = 1.0;
                    }
                    if (Lidar_BEV_map.at<cv::Vec3f>(y,x)[0] < new_point.z){
                        Lidar_BEV_map.at<cv::Vec3f>(y,x)[0] = new_point.z;
                    }
                    if (new_point.z < z_min){
                        z_min = new_point.z;
                    }
                }
            }
        }
    }

    for(geometry_msgs::Point32 point : pc_msg2->points){
        geometry_msgs::Point32 new_point;
        tf::Vector3 p = tf::Vector3(point.x, point.y, point.z);
        tf::Transform m = lidar_tf_2 * tf::Transform(id, p);
        tf::Vector3 v = m.getOrigin();
        new_point = vector_to_point(v);

        if (!((new_point.x >= 0) && (new_point.x <= 4.5) && (new_point.y >= -1) && (new_point.y <= 1)))
        {
            if ((new_point.x * map_scale + img_cor_y >= 0) && (new_point.x * map_scale + img_cor_y < map_height)){
                if((new_point.y * map_scale + img_cor_x >= 0) && (new_point.y * map_scale + img_cor_x < map_width)){
                    int y = (map_height - 1) - static_cast<int>(new_point.x * map_scale + img_cor_y);
                    int x = (map_width - 1) - static_cast<int>(new_point.y * map_scale + img_cor_x);
                    if (Lidar_BEV_map.at<cv::Vec3f>(y,x)[2] == 0.0){
                        Lidar_BEV_map.at<cv::Vec3f>(y,x)[2] = 1.0;
                    }
                    if (Lidar_BEV_map.at<cv::Vec3f>(y,x)[0] < new_point.z){
                        Lidar_BEV_map.at<cv::Vec3f>(y,x)[0] = new_point.z;
                    }
                    if (new_point.z < z_min){
                        z_min = new_point.z;
                    }
                }
            }
        }
    }

    double z_max = z_min + 3.5;
    
    for(int i = 0; i < map_width; i++){
        for(int j = 0; j < map_height; j++){
            if(Lidar_BEV_map.at<cv::Vec3f>(j,i)[2] == 1.0){
                Lidar_BEV_map.at<cv::Vec3f>(j,i)[0] = (Lidar_BEV_map.at<cv::Vec3f>(j,i)[0] - z_min) / (z_max - z_min);
                if(Lidar_BEV_map.at<cv::Vec3f>(j,i)[0] > 1.0){
                    Lidar_BEV_map.at<cv::Vec3f>(j,i)[0] = 1.0;
                }
            }
        }
    }
    //=================== End ===================//

    //=================== Camera Data ===================//

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img_msg1, sensor_msgs::image_encodings::BGR8);
    cv::warpPerspective(cv_ptr->image,Camera_BEV_front,homography_2, cv::Size(map_height, map_width));
    
    cv_ptr = cv_bridge::toCvCopy(img_msg2, sensor_msgs::image_encodings::BGR8);
    cv::warpPerspective(cv_ptr->image,Camera_BEV_right,homography_1, cv::Size(map_height, map_width));
    
    cv_ptr = cv_bridge::toCvCopy(img_msg3, sensor_msgs::image_encodings::BGR8);
    cv::warpPerspective(cv_ptr->image,Camera_BEV_left,homography_3, cv::Size(map_height, map_width));

    if(mask_gen == false){
        mask_gen = true;
        for(int i=0; i<map_height; i++)
        {
            for(int j=0; j<map_width; j++)
            {
                if(Camera_BEV_front.at<Vec3b>(j,i)[0] == 0 && Camera_BEV_front.at<Vec3b>(j,i)[1] == 0 && Camera_BEV_front.at<Vec3b>(j,i)[2] == 0){
                    front_mask.at<uchar>(j,i) = 0;
                }
                if(Camera_BEV_right.at<Vec3b>(j,i)[0] == 0 && Camera_BEV_right.at<Vec3b>(j,i)[1] == 0 && Camera_BEV_right.at<Vec3b>(j,i)[2] == 0){
                    right_mask.at<uchar>(j,i) = 0;
                }
                if(Camera_BEV_left.at<Vec3b>(j,i)[0] == 0 && Camera_BEV_left.at<Vec3b>(j,i)[1] == 0 && Camera_BEV_left.at<Vec3b>(j,i)[2] == 0){
                    left_mask.at<uchar>(j,i) = 0;
                }
            }
        }
    }

    Camera_BEV_right.copyTo(Image_BEV_map, right_mask);
    Camera_BEV_left.copyTo(Image_BEV_map, left_mask);
    Camera_BEV_front.copyTo(Image_BEV_map, front_mask);
    //=================== End ===================//

    //=================== Save Data ===================//
    stringstream save_path_object;
    save_path_object << ros::package::getPath("hmg_data_collector") << "/data/label/" << get_string(data_num,6) << ".txt";
    ofstream out_object(save_path_object.str());
    
    for(int i=0; i<output_data.size(); i++){
        out_object << output_data[i][0] << " " << output_data[i][1] << " " << output_data[i][2] << " " << output_data[i][3] << " " << output_data[i][4] << " " << output_data[i][5] << std::endl;
    }

    stringstream save_path_lidar;
    save_path_lidar << ros::package::getPath("hmg_data_collector") << "/data/" << "lidar/" << get_string(data_num,6) << ".png";
    cv::imwrite(save_path_lidar.str(), 255*Lidar_BEV_map);

    stringstream save_path_camera;
    save_path_camera << ros::package::getPath("hmg_data_collector") << "/data/" << "camera/" << get_string(data_num,6) << ".png";
    cv::imwrite(save_path_camera.str(), Image_BEV_map);

    data_num++;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start); 
    std::cout << "inference taken : " << duration.count() << " ms" << endl; 
    //=================== End ===================//

}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "hmg_data_collector");

    DataCollector data_collector;
    
    ros::spin();

    return 0;
}