CMAKE_MINIMUM_REQUIRED(VERSION 3.28)

PROJECT(HolaCuda)
INCLUDE(CheckLanguage)

OPTION(ENABLE_CUDA "Enable CUDA support" ON)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	ENABLE_LANGUAGE(CUDA)
	add_executable(hola src/hola.cu)
else()
	message(STATUS "No CUDA support")
endif()

