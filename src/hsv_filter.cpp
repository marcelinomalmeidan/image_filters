
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
  bool display_results_;
  int invert_;
  std::string input_topic_, out_topic_;
  std::vector<double> crop_percent_;

public:
  ImageConverter(ros::NodeHandle *nh) : it_(nh_){

    nh_ = *nh;
    // Get parameters
    bool invert;
    nh_.getParam("h_min", h_min_);
    nh_.getParam("h_max", h_max_);
    nh_.getParam("s_min", s_min_);
    nh_.getParam("s_max", s_max_);
    nh_.getParam("v_min", v_min_);
    nh_.getParam("v_max", v_max_);
    nh_.getParam("invert_result", invert);
    nh_.getParam("display_result", display_results_);
    nh_.getParam("input_topic", input_topic_);
    nh_.getParam("out_topic", out_topic_);
    nh_.getParam("crop_percent", crop_percent_);

    for (uint i = 0; i < 4; i++) {
      crop_percent_[i] = std::max(std::min(crop_percent_[i], 100.0), 0.0);
    }
    if (invert) {
      invert_ = 1;
    } else {
      invert_ = 0;
    }

    if (s_max_ < s_min_) std::swap(s_max_, s_min_);
    if (v_max_ < v_min_) std::swap(v_max_, v_min_);


    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(input_topic_, 1, &ImageConverter::image_callback, this);
    image_pub_ = it_.advertise(out_topic_, 1);

    ROS_INFO("Input RGB topic: %s", image_sub_.getTopic().c_str());
    ROS_INFO("Output topic: %s", image_pub_.getTopic().c_str());

    if(display_results_){
      int max_slider = 255;
      cv::namedWindow(OPENCV_WINDOW);
      cv::createTrackbar( "H min", OPENCV_WINDOW, &h_min_, max_slider, this->hmin_trackbar);
      cv::createTrackbar( "H max", OPENCV_WINDOW, &h_max_, max_slider, this->hmax_trackbar);
      cv::createTrackbar( "S min", OPENCV_WINDOW, &s_min_, max_slider, this->smin_trackbar);
      cv::createTrackbar( "S max", OPENCV_WINDOW, &s_max_, max_slider, this->smax_trackbar);
      cv::createTrackbar( "V min", OPENCV_WINDOW, &v_min_, max_slider, this->vmin_trackbar);
      cv::createTrackbar( "V max", OPENCV_WINDOW, &v_max_, max_slider, this->vmax_trackbar);
      cv::createTrackbar( "Invert result", OPENCV_WINDOW, &invert_, 1, this->invert_trackbar);
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

    // Make sure HSV values are in the right range
    this->check_hsv_values();

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

  void check_hsv_values() {
    if (h_min_ > h_max_) {
      h_min_ = h_max_;
    }
    if (s_min_ > s_max_) {
      s_min_ = s_max_;
    }
    if (v_min_ > v_max_) {
      v_min_ = v_max_;
    }
    if (h_max_ < h_min_) {
      h_max_ = h_min_;
    }
    if (s_max_ < s_min_) {
      s_max_ = s_min_;
    }
    if (v_max_ < v_min_) {
      v_max_ = v_min_;
    }
  }

  static void hmin_trackbar( int, void* ) { }
  static void smin_trackbar( int, void* ) { }
  static void vmin_trackbar( int, void* ) { }
  static void hmax_trackbar( int, void* ) { }
  static void smax_trackbar( int, void* ) { }
  static void vmax_trackbar( int, void* ) { }
  static void invert_trackbar( int, void* ) { }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "hsv_filter");
  ros::NodeHandle node("~");
  ImageConverter ic(&node);

  ros::spin();

  return 0;
}