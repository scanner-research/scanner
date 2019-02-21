from attr import attrs, attrib
from scannerpy.common import FrameType, ScannerException
from scannerpy.protobufs import protobufs
import pickle
from typing import NewType, Generic, TypeVar, Any
import numpy as np
import struct

PYTHON_TYPE_REGISTRY = {}

@attrs(frozen=True)
class ScannerTypeInfo:
    type = attrib()
    cpp_name = attrib()
    serializer = attrib()
    deserializer = attrib()

def _register_type(ty, cpp_name, serializer, deserializer):
    global PYTHON_TYPE_REGISTRY
    PYTHON_TYPE_REGISTRY[ty] = ScannerTypeInfo(
        type=ty,
        cpp_name=cpp_name,
        serializer=serializer,
        deserializer=deserializer)

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
    _register_type(cls, name, cls.serializer, cls.deserializer)
    return cls


def ProtoList(name, proto):
    def serializer(proto_list):
        s = struct.pack('=Q', len(proto_list))
        for element in proto_list:
            serialized = element.SerializeToString()
            s += struct.pack('=Q', len(serialized))
            s += serialized
        return s

    def deserializer(buf):
        (N, ) = struct.unpack("=Q", buf[:8])
        buf = buf[8:]
        elements = []
        for i in range(N):
            (serialized_size, ) = struct.unpack("=Q", buf[:8])
            buf = buf[8:]
            element = proto()
            element.ParseFromString(buf[:serialized_size])
            buf = buf[serialized_size:]
            elements.append(element)
        return element

    return register_type(type(name, (), dict(serializer=serializer, deserializer=deserializer)))

BboxList = ProtoList('BboxList', protobufs.BoundingBox)

@register_type
class Histogram:
    def serializer(buf):
        raise NotImplemented

    def deserializer(buf):
        return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)
