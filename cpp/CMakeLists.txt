# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only
#
# C++ vertical slice. Until milestone M5 this tree deliberately depends on
# neither rclib's C++ core nor rtctrl; it exists so the build, test, and CI
# path is qualified before real inference code arrives.
cmake_minimum_required(VERSION 3.22)
project(arm_rc_ctrl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ARM_RC_CTRL_BUILD_TESTS "Build the Catch2 test suite" ON)
option(ARM_RC_CTRL_WERROR "Treat compiler warnings in project sources as errors" OFF)

add_library(arm_rc_ctrl_warnings INTERFACE)
target_compile_options(arm_rc_ctrl_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow>
  $<$<AND:$<BOOL:${ARM_RC_CTRL_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>)

add_library(arm_rc_ctrl src/version.cpp)
add_library(arm_rc_ctrl::arm_rc_ctrl ALIAS arm_rc_ctrl)
target_compile_features(arm_rc_ctrl PUBLIC cxx_std_17)
target_include_directories(arm_rc_ctrl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(arm_rc_ctrl PRIVATE ARM_RC_CTRL_VERSION="${PROJECT_VERSION}")
target_link_libraries(arm_rc_ctrl PRIVATE arm_rc_ctrl_warnings)

add_subdirectory(apps)

if(ARM_RC_CTRL_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  # Catch2 v3.7.1, pinned by the immutable commit the tag resolves to.
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG fa43b77429ba76c462b1898d6cd2f2d7a9416b14)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  add_subdirectory(tests)
endif()
