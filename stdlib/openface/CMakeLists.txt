find_package(OpenFace REQUIRED)

set(SOURCE_FILES openface_kernel.cpp)

add_library(openface OBJECT ${SOURCE_FILES})

message(STATUS "${OpenFace_LIBRARIES}")

target_include_directories(openface PUBLIC "${OpenFace_INCLUDE_DIRS}")
list(APPEND STDLIB_LIBRARIES "${OpenFace_LIBRARIES}")
set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)

list(APPEND OPENCV_COMPONENTS core objdetect)
set(OPENCV_COMPONENTS ${OPENCV_COMPONENTS} PARENT_SCOPE)
