# Scanner outputs build flags that add Scanner to your library and include paths
# as well as cover common flags for building a shared library.
SCANNER_FLAGS = `python -c "import scannerpy.stdlib.build_flags"`

# Change this to whatever your OpenCV directory is
OpenCV_DIR=/opt/opencv-3.2.0

all: build/libresize_op.so

clean:
	rm -rf build/*

# protoc generates the Python and C++ bindings for the ResizeArgs class.
build/resize.pb.cc: resize.proto
	protoc resize.proto --python_out=build --cpp_out=build

# g++ builds the op library
build/libresize_op.so: build/resize.pb.cc resize_op.cpp
	g++ $^ -o $@ ${SCANNER_FLAGS} \
		-I ${OpenCV_DIR}/include -L ${OpenCV_DIR}/lib -lopencv_core -lopencv_imgproc \
		-I build

.PHONY: clean
