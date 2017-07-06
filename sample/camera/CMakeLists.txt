cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../movie/cmake)

message(STATUS ${CMAKE_MODULE_PATH})

FIND_PACKAGE(CUDA REQUIRED)

find_package(OpenCV REQUIRED core) # dummy
if(OpenCV_VERSION VERSION_LESS 3.0) 
	find_package(OpenCV REQUIRED core highgui imgproc)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)

find_package(GLFW REQUIRED)

include_directories(../../include)
include_directories(../movie)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLFW_INCLUDE_DIR})

CUDA_ADD_EXECUTABLE(stereo_camera
stereosgm_camera.cpp
../movie/demo.cpp
../movie/renderer.cpp
../movie/cuda_gl.cu
)

TARGET_LINK_LIBRARIES(stereo_camera sgm  ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${OpenCV_LIBS} ${GLEW_LIBRARIES} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
