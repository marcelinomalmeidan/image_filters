cmake_minimum_required(VERSION 2.8.3)
project(image_filters)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  message_filters
)

find_package(OpenCV REQUIRED)


catkin_package(

)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(hsv_filter src/hsv_filter.cpp)
target_link_libraries(hsv_filter ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(depth_filter src/depth_filter.cpp)
target_link_libraries(depth_filter ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})