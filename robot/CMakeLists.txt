cmake_minimum_required(VERSION 3.3)
project(robot)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        robot.c bridge.c bridge.h)

add_executable(robot ${SOURCE_FILES})
target_link_libraries(robot pthread)