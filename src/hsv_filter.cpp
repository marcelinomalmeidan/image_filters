
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


static const std::string OPENCV_WINDOW = "Image window";


class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  int h_min_, h_max_, s_min_, s_max_, v_min_, v_max_;
  cv::Scalar lower_color_range_, upper_color_range_;
  bool invert_, display_results_;
  std::string input_topic_, out_topic_;
  std::vector<double> crop_percent_;

public:
  ImageConverter(ros::NodeHandle *nh) : it_(nh_){

    nh_ = *nh;
    // Get parameters
    nh_.getParam("h_min", h_min_);
    nh_.getParam("h_max", h_max_);
    nh_.getParam("s_min", s_min_);
    nh_.getParam("s_max", s_max_);
    nh_.getParam("v_min", v_min_);
    nh_.getParam("v_max", v_max_);
    nh_.getParam("invert_result", invert_);
    nh_.getParam("display_result", display_results_);
    nh_.getParam("input_topic", input_topic_);
    nh_.getParam("out_topic", out_topic_);
    nh_.getParam("crop_percent", crop_percent_);

    ROS_INFO("Input topic: %s", input_topic_.c_str());
    ROS_INFO("Output topic: %s", out_topic_.c_str());
    for (uint i = 0; i < 4; i++) {
      crop_percent_[i] = std::max(std::min(crop_percent_[i], 100.0), 0.0);
    }

    if (s_max_ < s_min_) std::swap(s_max_, s_min_);
    if (v_max_ < v_min_) std::swap(v_max_, v_min_);


    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(input_topic_, 1, &ImageConverter::image_callback, this);
    image_pub_ = it_.advertise(out_topic_, 1);

    if(display_results_){
      cv::namedWindow(OPENCV_WINDOW);
    }
  }

  ~ImageConverter() {
    if(display_results_){
      cv::destroyWindow(OPENCV_WINDOW);
    }
  }

  void image_callback(const sensor_msgs::ImageConstPtr& msg) {
    // Convert to Opencv
    cv::Mat input_image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
    
    // Transform image into HSV
    cv::Mat hsv_image;
    cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);

    // Get crop info
    uint height = msg->height;
    uint width = msg->width;
    uint roi_left = std::floor(crop_percent_[0]*float(width)/100.0);
    uint roi_bottom = std::floor(crop_percent_[1]*float(height)/100.0);
    uint roi_right = std::floor(crop_percent_[2]*float(width)/100.0);
    uint roi_up = std::floor(crop_percent_[3]*float(height)/100.0);
    uint roi_width = width - roi_left - roi_right;
    uint roi_height = height - roi_up - roi_bottom;
    cv::Rect ROI(roi_left, roi_up, roi_width, roi_height);
    cv::Mat roi_mask(hsv_image.size(), CV_8UC1, cv::Scalar::all(0));
    roi_mask(ROI).setTo(cv::Scalar::all(255));

    // Filter image
    cv::Mat mask, output_image;
    cv::inRange(hsv_image, cv::Scalar(h_min_, s_min_, v_min_), 
                cv::Scalar(h_max_, s_max_, v_max_), mask);
    if(invert_) {
      mask = cv::Scalar::all(255) - mask;
    }
    cv::bitwise_and(mask, roi_mask, mask);
    hsv_image.copyTo(output_image, mask);
    cv::cvtColor(output_image, output_image, cv::COLOR_HSV2BGR);

    // Publish the image.
    sensor_msgs::Image::Ptr out_img = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, output_image).toImageMsg();
    image_pub_.publish(out_img);

    if (display_results_) {
      cv::imshow(OPENCV_WINDOW, output_image);
      cv::waitKey(1);
    }

  }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "hsv_filter");
  ros::NodeHandle node("~");
  ImageConverter ic(&node);

  ros::spin();

  return 0;
}