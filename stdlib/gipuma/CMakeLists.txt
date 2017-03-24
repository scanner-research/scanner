find_package(Gipuma REQUIRED)

set(SOURCE_FILES gipuma_kernel.cpp)

add_library(gipuma OBJECT ${SOURCE_FILES})

target_include_directories(gipuma PUBLIC "${GIPUMA_INCLUDE_DIRS}")
list(APPEND STDLIB_LIBRARIES "${GIPUMA_LIBRARIES}")

set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)
