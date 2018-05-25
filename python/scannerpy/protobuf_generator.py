import os.path
import imp
import sys

from scannerpy.common import *

import scannerpy._python as bindings
import scanner.stdlib.stdlib_pb2 as stdlib_types
import scanner.metadata_pb2 as metadata_types
import scanner.source_args_pb2 as source_types
import scanner.sink_args_pb2 as sink_types
import scanner.sampler_args_pb2 as sampler_types
import scanner.engine.rpc_pb2 as rpc_types
import scanner.engine.rpc_pb2_grpc as grpc_types
import scanner.types_pb2 as misc_types
from google.protobuf.descriptor import FieldDescriptor


class ProtobufGenerator:
    def __init__(self, cfg):
        self._mods = []
        for mod in [
                misc_types, rpc_types, grpc_types, metadata_types,
                source_types, sink_types, sampler_types, stdlib_types
        ]:
            self.add_module(mod)

    def add_module(self, path):
        if isinstance(path, str):
            if not os.path.isfile(path):
                raise ScannerException('Protobuf path does not exist: {}'
                                       .format(path))
            imp.acquire_lock()
            mod = imp.load_source('_ignore', path)
            imp.release_lock()
        else:
            mod = path
        self._mods.append(mod)

    def __getattr__(self, name):
        for mod in self._mods:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise ScannerException('No protobuf with name {}'.format(name))


def python_to_proto(protos, proto_name, obj):
    args_proto = getattr(protos, proto_name)

    def analyze_proto(p):
        fields = {}
        for f in p.fields:
            child_fields = None
            if f.type == FieldDescriptor.TYPE_MESSAGE:
                child_fields = analyze_proto(f.message_type)

            fields[f.name] = {
                'type':
                f.type,
                'message':
                getattr(protos, f.message_type.name)
                if f.message_type is not None else None,
                'repeated':
                f.label == FieldDescriptor.LABEL_REPEATED,
                'fields':
                child_fields,
            }
        return fields

    p = analyze_proto(args_proto.DESCRIPTOR)

    def create_obj(proto, p, obj):
        if isinstance(obj, proto):
            return obj
        elif not isinstance(obj, dict):
            raise ScannerException('Attempted to bind a non-dict type to a '
                                   'protobuf')

        proto_obj = proto()

        for k, v in obj.items():
            if k not in p:
                raise ScannerException(
                    'Protobuf {} does not have field {:s}'.format(
                        proto_name, k))
            desc = p[k]

            # If a message field
            def make_field(val):
                if desc['type'] == FieldDescriptor.TYPE_MESSAGE:
                    # If a message field, we need to recursively invoke
                    # serialization
                    return create_obj(desc['message'], desc['fields'], val)
                else:
                    return val

            if p[k]['repeated']:
                # If a repeated field, we need to set using slicing
                data = []
                for vi in v:
                    data.append(make_field(vi))
                getattr(proto_obj, k)[:] = data
            elif p[k]['message'] is not None:
                # If a message field, have to CopyFrom, can't use direct assignment
                getattr(proto_obj, k).CopyFrom(make_field(v))
            else:
                # Just set the regular field
                setattr(proto_obj, k, make_field(v))

        return proto_obj

    return create_obj(args_proto, p, obj).SerializeToString()
