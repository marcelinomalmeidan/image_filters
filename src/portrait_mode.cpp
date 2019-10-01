
#include "ros/ros.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


static const std::string OPENCV_WINDOW = "Image window";
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicyApprox;
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Sync;



class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub_, depth_sub_;
  image_transport::Publisher image_pub_;
  float th_min_, th_max_;     // Thresholds in meters
  int th_min_mm_, th_max_mm_; // Thresholds in millimeters
  float depth_conversion_;
  bool invert_, display_results_, is_rgb_;
  std::string in_rgb_topic_, in_depth_topic_, out_topic_;
  std::string output_type_;
  std::vector<double> crop_percent_;
  boost::shared_ptr<Sync> sync_;
  int smooth_factor_, smooth_factor_max_;
  int mask_dilation_size_, mask_dilation_size_max_;
  // message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync_(rgb_sub_, depth_sub_, 10);

public:
  ImageConverter(ros::NodeHandle *nh) : it_(nh_) {

    nh_ = *nh;
    // Get parameters
    nh_.getParam("threshold_min", th_min_);
    nh_.getParam("threshold_max", th_max_);
    nh_.getParam("mask_dilation_size", mask_dilation_size_);
    nh_.getParam("depth_conversion", depth_conversion_);
    nh_.getParam("invert_result", invert_);
    nh_.getParam("display_result", display_results_);
    nh_.getParam("input_rgb_topic", in_rgb_topic_);
    nh_.getParam("input_depth_topic", in_depth_topic_);
    nh_.getParam("out_topic", out_topic_);
    nh_.getParam("crop_percent", crop_percent_);
    nh_.getParam("smooth_factor", smooth_factor_);
    nh_.getParam("output_type", output_type_);
    
    // Setting crop bounds
    for (uint i = 0; i < 4; i++) {
      crop_percent_[i] = std::max(std::min(crop_percent_[i], 100.0), 0.0);
    }

    // Conversion from meters to millimiters
    th_min_mm_ = std::floor(th_min_*1000);
    th_max_mm_ = std::floor(th_max_*1000);

    // Select output type
    if (output_type_.compare("GRAY") == 0) {
      ROS_INFO("Output is GRAY!");
      is_rgb_ = false;
    } else if (output_type_.compare("RGB") == 0) {
      ROS_INFO("Output is RGB!");
      is_rgb_ = true;
    } else {
      ROS_WARN("Output types are GRAY or RGB. Output being set to RGB!");
      is_rgb_ = true;
    }

    // Set max values
    smooth_factor_max_ = 30;
    mask_dilation_size_max_ = 20;

    // Subscribe to input video feed and publish output video feed
    rgb_sub_.subscribe(nh_, in_rgb_topic_, 1);
    depth_sub_.subscribe(nh_, in_depth_topic_, 1);
    sync_.reset(new Sync(SyncPolicy(10), rgb_sub_, depth_sub_));
    sync_->registerCallback(boost::bind(&ImageConverter::image_callback, this, _1, _2));
    image_pub_ = it_.advertise(out_topic_, 1);

    ROS_INFO("Input RGB topic: %s", rgb_sub_.getTopic().c_str());
    ROS_INFO("Input Depth topic: %s", depth_sub_.getTopic().c_str());
    ROS_INFO("Output topic: %s", image_pub_.getTopic().c_str());

    if(display_results_) {
      int max_dist = 10000;
      cv::namedWindow(OPENCV_WINDOW);
      cv::createTrackbar( "Smoothness", OPENCV_WINDOW, &smooth_factor_, smooth_factor_max_, this->smooth_trackbar);
      cv::createTrackbar( "Mask Dilation Size", OPENCV_WINDOW, &mask_dilation_size_, mask_dilation_size_max_, this->dilation_trackbar);
      cv::createTrackbar( "Min Distance Threshold (mm)", OPENCV_WINDOW, &th_min_mm_, max_dist, this->min_th_trackbar);
      cv::createTrackbar( "Max Distance Threshold (mm)", OPENCV_WINDOW, &th_max_mm_, max_dist, this->max_th_trackbar);
    }
  }

  ~ImageConverter() {
    if(display_results_){
      cv::destroyWindow(OPENCV_WINDOW);
    }
  }

  void image_callback(const sensor_msgs::ImageConstPtr& rgb_msg,
                      const sensor_msgs::ImageConstPtr& depth_msg) {

    if((rgb_msg->width != depth_msg->width) || (rgb_msg->height != depth_msg->height)) {
      ROS_WARN("Depth image size does not match color image size! Callback will not execute!");
      return;
    }

    // Convert to Opencv
    cv::Mat input_image;
    if (is_rgb_) {
      input_image = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8)->image;
    } else {
      input_image = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8)->image;
      cv::cvtColor(input_image,input_image,CV_BGR2GRAY);
    }
    cv::Mat depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
 
    // Get crop info
    uint height = rgb_msg->height;
    uint width = rgb_msg->width;
    uint roi_left = std::floor(crop_percent_[0]*float(width)/100.0);
    uint roi_bottom = std::floor(crop_percent_[1]*float(height)/100.0);
    uint roi_right = std::floor(crop_percent_[2]*float(width)/100.0);
    uint roi_up = std::floor(crop_percent_[3]*float(height)/100.0);
    uint roi_width = width - roi_left - roi_right;
    uint roi_height = height - roi_up - roi_bottom;
    cv::Rect ROI(roi_left, roi_up, roi_width, roi_height);
    cv::Mat roi_mask(input_image.size(), CV_8UC1, cv::Scalar::all(0));
    roi_mask(ROI).setTo(cv::Scalar::all(255));

    cv::Mat output_image(input_image.size(), CV_8UC1, cv::Scalar::all(0));
    if (smooth_factor_ > 0) {
      // Perform depth filtering in the RGB image
      GetMinMaxThreshold();
      cv::Mat depth_mask(input_image.size(), CV_8UC1, cv::Scalar::all(255));
      for (uint i = 0; i < height; i++) {
        for (uint j = 0; j < width; j++) {
          cv::Scalar intensity = depth_image.at<float>(i,j);
          float pixel_depth = intensity[0]*depth_conversion_;
          if ((pixel_depth < th_min_) || (pixel_depth > th_max_) || isnan(pixel_depth)) {
            depth_mask.at<uchar>(i,j) = 0;
          }
        }
      }

      // Dilate mask
      cv::Mat depth_mask_dilated(input_image.size(), CV_8UC1, cv::Scalar::all(255));
      int dilation_size = mask_dilation_size_;
      cv::Mat element = getStructuringElement(cv::MORPH_CROSS,
                      cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                      cv::Point(dilation_size, dilation_size) );
      cv::dilate(depth_mask, depth_mask_dilated, element);

      // Get backrgound and blur it
      cv::Mat inv_mask = cv::Scalar::all(255) - depth_mask_dilated;
      this->MaskedSmoothOptimised(input_image, inv_mask, output_image, smooth_factor_);
    } else {
      output_image = input_image;
    }

    // Publish the image.
    sensor_msgs::Image::Ptr out_img;
    if (is_rgb_) {
      out_img = cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::BGR8, output_image).toImageMsg();
    } else {
      out_img = cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::MONO8, output_image).toImageMsg();
    }
    // sensor_msgs::Image::Ptr out_img = cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::TYPE_32FC3, output_image).toImageMsg();
    // sensor_msgs::Image::Ptr out_img = cv_bridge::CvImage(rgb_msg->header, sensor_msgs::image_encodings::TYPE_8UC1, output_image).toImageMsg();
    image_pub_.publish(out_img);

    if (display_results_) {
      cv::imshow(OPENCV_WINDOW, output_image);
      cv::waitKey(1);
    }

  }

  bool MaskedSmoothOptimised(cv::Mat mSrc, cv::Mat mMask, cv::Mat &mDst, double smooth_factor) {
    if(mSrc.empty())
    {
        return 0;
    }

    if (is_rgb_) {
      cv::Mat mGSmooth;
      cv::cvtColor(mMask, mMask, cv::COLOR_GRAY2BGR);

      mDst = cv::Mat(mSrc.size(), CV_32FC3);
      mMask.convertTo(mMask, CV_32FC3, 1.0/255.0);
      mSrc.convertTo(mSrc, CV_32FC3,1.0/255.0);

      cv::blur(mSrc,mGSmooth, cv::Size(smooth_factor, smooth_factor), cv::Point(-1,-1)); 
      cv::blur(mMask,mMask, cv::Size(smooth_factor, smooth_factor), cv::Point(-1,-1));   

      cv::Mat M1,M2,M3;

      cv::subtract(cv::Scalar::all(1.0),mMask,M1);
      cv::multiply(M1,mSrc,M2);
      cv::multiply(mMask,mGSmooth,M3);
      cv::add(M2,M3,mDst);

      mDst.convertTo(mDst, CV_8UC3, 255);
    } else {
      mDst = cv::Mat(mSrc.size(), CV_32FC1);
      mMask.convertTo(mMask, CV_32FC1, 1.0/255.0);
      mSrc.convertTo(mSrc, CV_32FC1,1.0/255.0);

      cv::Mat mGSmooth; 
      cv::blur(mSrc,mGSmooth, cv::Size(smooth_factor, smooth_factor), cv::Point(-1,-1)); 
      cv::blur(mMask,mMask, cv::Size(smooth_factor, smooth_factor), cv::Point(-1,-1));

      cv::Mat M1,M2,M3;

      cv::subtract(cv::Scalar::all(1.0),mMask,M1);
      cv::multiply(M1,mSrc,M2);
      cv::multiply(mMask,mGSmooth,M3);
      cv::add(M2,M3,mDst);
      mDst.convertTo(mDst, CV_8UC1, 255);  
    }

    return true;
  }

  void GetMinMaxThreshold() {
    th_min_ = float(th_min_mm_)/1000.0;
    th_max_ = float(th_max_mm_)/1000.0;
  }

  static void smooth_trackbar( int, void* ) { }
  static void dilation_trackbar( int, void* ) { }
  static void min_th_trackbar( int, void* ) { }
  static void max_th_trackbar( int, void* ) { }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "portrait_mode");
  ros::NodeHandle node("~");
  ImageConverter ic(&node);

  ros::spin();

  return 0;
}