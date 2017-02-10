# Scanner outputs build flags that add Scanner to your library and include paths
# as well as cover common flags for building a shared library.
SCANNER_FLAGS = `python -c "import scannerpy; print(scannerpy.Database().get_build_flags())"`

all:
# protoc generates the Python and C++ bindings for the ResizeArgs class.
	protoc args.proto --python_out=. --cpp_out=.
#	g++ builds the op library, making sure to include the generated protobuf
# file and link in libprotobuf.
	g++ args.pb.cc resize_op.cpp -o resize_op.so ${SCANNER_FLAGS} -lprotobuf

clean:
	rm -f args_pb2.py args_pb2.pyc args.pb.cc args.pb.h resize_op.so
