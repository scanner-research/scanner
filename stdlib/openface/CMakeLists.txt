find_package(OpenFace REQUIRED)

set(SOURCE_FILES openface_kernel.cpp)

add_library(openface OBJECT ${SOURCE_FILES})

target_include_directories(openface PUBLIC "${OpenFace_INCLUDE_DIRS}")
list(APPEND STDLIB_LIBRARIES "${OpenFace_LIBRARIES}")

set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)
