cmake_minimum_required(VERSION 3.0.2)
project(hmg_data_collector)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  sensor_msgs
  message_generation
  message_filters
  std_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  tf_conversions
  tf
  tf2
  hellocm_msgs
)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)

# add_message_files(
#   FILES 
#   ObservPoint.msg
#   ObjectSensorObj.msg
#   ObjectSensor.msg
# )

# generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

catkin_package(
	INCLUDE_DIRS include
	CATKIN_DEPENDS roscpp std_msgs sensor_msgs

	DEPENDS
	opencv_highgui opencv_imgproc
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
include_directories(include)

add_executable(hmg_data_collector src/main.cpp)
add_dependencies(hmg_data_collector ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(hmg_data_collector ${OpenCV_LIBS} ${catkin_LIBRARIES})