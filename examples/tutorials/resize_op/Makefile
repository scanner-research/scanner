# Scanner outputs build flags that add Scanner to your library and include paths
# as well as cover common flags for building a shared library.
SCANNER_CFLAGS = `python3 -c "import scannerpy.build_flags as b; b.print_compile_flags()"`
SCANNER_LD_FLAGS = `python3 -c "import scannerpy.build_flags as b; b.print_link_flags()"`

# Change this to wherever you installed deps.sh to
INSTALL_DIR=../../../thirdparty/install

all: build/libresize_op.so

clean:
	rm -rf build/*

# protoc generates the Python and C++ bindings for the ResizeArgs class.
build/resize.pb.cc: resize.proto
	protoc resize.proto --python_out=build --cpp_out=build

# g++ builds the op library
build/libresize_op.so: build/resize.pb.cc resize_op.cpp
	g++ -shared -fPIC $^ -o $@ ${SCANNER_CFLAGS} ${SCANNER_LD_FLAGS} \
		-I ${INSTALL_DIR}/include -I build \
		-L ${INSTALL_DIR}/lib -lprotobuf -lglog -lopencv_core -lopencv_imgproc

.PHONY: clean
