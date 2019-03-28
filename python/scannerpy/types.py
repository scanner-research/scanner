from attr import attrs, attrib
from scannerpy.common import ScannerException
from scannerpy.protobufs import protobufs
import pickle
from typing import NewType, Generic, TypeVar, Any
import numpy as np
import struct

PYTHON_TYPE_REGISTRY = {}

# This is a special built-in type for video frame streams.
# Class purely for type annotation.
class FrameType(object):
    pass

BlobType = bytes

@attrs(frozen=True)
class ScannerTypeInfo:
    type = attrib()
    cpp_name = attrib()
    serialize = attrib()
    deserialize = attrib()

def _register_type(ty, cpp_name, serialize, deserialize):
    global PYTHON_TYPE_REGISTRY
    PYTHON_TYPE_REGISTRY[ty] = ScannerTypeInfo(
        type=ty,
        cpp_name=cpp_name,
        serialize=serialize,
        deserialize=deserialize)

def get_type_info(ty):
    global PYTHON_TYPE_REGISTRY
    if not ty in PYTHON_TYPE_REGISTRY:
        raise ScannerException("Type `{}` has not been registered with Scanner".format(ty.__name__))
    return PYTHON_TYPE_REGISTRY[ty]

def get_type_info_cpp(cpp_name):
    global PYTHON_TYPE_REGISTRY
    for ty in PYTHON_TYPE_REGISTRY.values():
        if ty.cpp_name == cpp_name:
            return ty
    raise ScannerException("Type `{}` has not been registered with Scanner".format(cpp_name))

_register_type(bytes, "Bytes", lambda x: x, lambda x: x)
_register_type(Any, "Any", pickle.dumps, pickle.loads)
_register_type(FrameType, "FrameType", lambda x: x, lambda x: x)

# TODO: document this
def register_type(cls):
    name = cls.__name__
    _register_type(cls, name, cls.serialize, cls.deserialize)
    return cls

def ProtobufType(name, proto):
    def serialize(proto_obj):
        return proto_obj.SerializeToString()

    def deserialize(buf):
        p = proto()
        p.ParseFromString(buf)
        return p

    return register_type(type(name, (), dict(serialize=serialize, deserialize=deserialize)))


def VariableList(name, typ):
    def serialize(variable_list):
        s = struct.pack('=Q', len(variable_list))
        for element in variable_list:
            serialized = typ.serialize(element)
            s += struct.pack('=Q', len(serialized))
            s += serialized
        return s

    def deserialize(buf):
        (N, ) = struct.unpack("=Q", buf[:8])
        buf = buf[8:]
        elements = []
        for i in range(N):
            (serialized_size, ) = struct.unpack("=Q", buf[:8])
            buf = buf[8:]
            element = typ.deserialize(buf[:serialized_size])
            buf = buf[serialized_size:]
            elements.append(element)
        return elements

    return register_type(type(name, (), dict(serialize=serialize, deserialize=deserialize)))

def UniformList(name, typ, size=None, parts=None):
    assert (size is not None) ^ (parts is not None)

    def serialize(uniform_list):
        return b''.join([typ.serialize(obj) for obj in uniform_list])

    def deserialize(buf):
        nonlocal size
        nonlocal parts

        # HACK(will): need a placeholder for an empty list
        if len(buf) <= 4:
            return []

        if parts is not None:
            size = len(buf) // parts

        assert len(buf) % size == 0
        return [typ.deserialize(buf[i:i+size]) for i in range(0, len(buf), size)]

    return register_type(type(name, (), dict(serialize=serialize, deserialize=deserialize)))

Bbox = ProtobufType('Bbox', protobufs.BoundingBox)
BboxList = VariableList('BboxList', Bbox)

@register_type
class NumpyArrayFloat32:
    def serialize(array):
        return array.tobytes()

    def deserialize(data_buffer):
        return np.frombuffer(data_buffer, dtype=np.float32)

@register_type
class NumpyArrayInt32:
    def serialize(array):
        return array.tobytes()

    def deserialize(data_buffer):
        return np.frombuffer(data_buffer, dtype=np.int32)

Histogram = UniformList('Histogram', NumpyArrayInt32, parts=3)

@register_type
class Image:
    def serialize(image):
        import cv2
        return cv2.imencode('.png', image)

    def deserialize(encoded_image):
        import cv2
        return cv2.imdecode(np.frombuffer(encoded_image, dtype=np.dtype(np.uint8)), cv2.IMREAD_COLOR)
