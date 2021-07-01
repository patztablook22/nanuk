cmake_minimum_required(VERSION 3.20)

project(nanuk)
set(CMAKE_CXX_STANDARD 14)

add_library(${PROJECT_NAME} SHARED
  nanuk.cpp
  neuron.cpp
  functions.cpp
  file_io.cpp
)

add_executable(example example.cpp)
target_link_libraries(example PUBLIC nanuk)
