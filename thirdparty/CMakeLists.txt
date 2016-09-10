# Copyright 2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)

project(LightscanThirdparty)

include(ExternalProject)

ExternalProject_Add(TinyToml
  GIT_REPOSITORY "https://github.com/mayah/tinytoml.git"
  GIT_TAG "3559856002eee57693349b8a2d8a0cf6250d269c"

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/tinytoml"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""

  INSTALL_COMMAND ""
)

ExternalProject_Add(JsonCpp
  GIT_REPOSITORY "https://github.com/open-source-parsers/jsoncpp.git"
  GIT_TAG "1.7.2"

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/jsoncpp"
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/jsoncpp
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
)

# storehouse
if (NOT OPENSSL_ROOT_DIR)
  set(OPENSSL_ROOT_DIR $ENV{OPENSSL_ROOT_DIR})
endif()

ExternalProject_Add(Storehouse
  GIT_REPOSITORY "https://github.com/apoms/storehouse"
  #GIT_TAG 0a439623f75c029912728d80cb7f1b8b48739ca4

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/storehouse"

  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/storehouse
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DBZIP2_INCLUDE_DIR=${BZIP2_INCLUDE_DIR}
      -DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}
      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
      -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}

  CMAKE_CACHE_ARGS
      -DGLOG_ROOT_DIR:FILEPATH=${GLOG_ROOT_DIR}
      -DGFLAGS_ROOT_DIR:FILEPATH=${GFLAGS_ROOT_DIR}
      -DBOOST_ROOT:FILEPATH=${BOOST_ROOT}
      -DCURL_INCLUDE_DIR:FILEPATH=${CURL_INCLUDE_DIR}
      -DCURL_LIBRARY:FILEPATH=${CURL_LIBRARY}
      -DCURL_LIBRARIES:FILEPATH=${CURL_LIBRARIES}

  INSTALL_DIR "${GLOBAL_OUTPUT_PATH}/storehouse"
)


# googletest
ExternalProject_Add(GoogleTest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG 0a439623f75c029912728d80cb7f1b8b48739ca4

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/googletest"

  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/googletest
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}

  INSTALL_DIR "${GLOBAL_OUTPUT_PATH}/googletest"
)
