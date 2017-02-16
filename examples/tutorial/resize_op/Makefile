# Scanner outputs build flags that add Scanner to your library and include paths
# as well as cover common flags for building a shared library.
SCANNER_FLAGS = `python -c "import scannerpy; scannerpy.Database().print_build_flags()"`

# Change this to whatever your OpenCV directory is
OpenCV_DIR=/opt/opencv-3.2.0

all:
# protoc generates the Python and C++ bindings for the ResizeArgs class.
	protoc args.proto --python_out=build --cpp_out=build
# g++ builds the op library
	g++ build/args.pb.cc resize_op.cpp -o build/libresize_op.so ${SCANNER_FLAGS} \
		-I ${OpenCV_DIR}/include -L ${OpenCV_DIR}/lib -lopencv_core -lopencv_imgproc \
		-I build
