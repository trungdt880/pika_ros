cmake_minimum_required(VERSION 3.8)
project(pika_nero_teleop)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/record_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
