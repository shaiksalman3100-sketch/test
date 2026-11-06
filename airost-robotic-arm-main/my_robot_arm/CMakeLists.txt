cmake_minimum_required(VERSION 3.8)
project(my_robot_arm)
find_package(ament_cmake REQUIRED)
install(
  DIRECTORY config launch meshes urdf env_assets
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
ament_package()
