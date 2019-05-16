cmake_minimum_required(VERSION 2.8.3)
project(image_filters)

option(USE_CUDA "Compile Cuda-enabled solutions" FALSE) # TRUE/FALSE

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

if(USE_CUDA)
  find_package(OpenCV 3.4.0 REQUIRED
      NO_MODULE
      PATHS /usr/local
      NO_DEFAULT_PATH
  )
else()
  find_package(OpenCV REQUIRED)
endif(USE_CUDA)

catkin_package(

)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

message("custom message")
message(WARNING ${OpenCV_INCLUDE_DIRS})
message(WARNING ${OpenCV_LIBRARIES})

add_executable(hsv_filter src/hsv_filter.cpp)
target_link_libraries(hsv_filter ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(depth_filter src/depth_filter.cpp)
target_link_libraries(depth_filter ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(portrait_mode src/portrait_mode.cpp)
target_link_libraries(portrait_mode ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


if(USE_CUDA)
  SET(OPENCV_CUDA_SAMPLES_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui
                                        opencv_ml opencv_video opencv_objdetect opencv_features2d
                                        opencv_calib3d opencv_superres
                                        opencv_cudaarithm opencv_cudafilters opencv_cudawarping opencv_cudaimgproc
                                        opencv_cudafeatures2d opencv_cudaoptflow opencv_cudabgsegm
                                        opencv_cudastereo opencv_cudalegacy opencv_cudaobjdetect)

  add_executable(portrait_mode_cuda src/portrait_mode_cuda.cpp)
  target_link_libraries(portrait_mode_cuda ${OpenCV_LIBRARIES} ${OPENCV_CUDA_SAMPLES_REQUIRED_DEPS} ${catkin_LIBRARIES})
endif(USE_CUDA)
unset(USE_CUDA CACHE)
