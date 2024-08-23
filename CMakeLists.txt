cmake_minimum_required(VERSION 3.28)

project(HolaCuda)

OPTION(ENABLE_CUDA "Enable CUDA support" ON)

CHECK_LANGUAGE(CUDA)
if(CMAKE_CUDA_COMPILER)
	ENABLE_LANGUAGE(CUDA)
else()
	message(STATUS "No CUDA support")
endif()

