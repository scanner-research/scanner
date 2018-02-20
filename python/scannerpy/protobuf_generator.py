from __future__ import absolute_import, division, print_function, unicode_literals
import os.path
import imp
import sys

from scannerpy.common import *

import scanner.stdlib.stdlib_pb2 as stdlib_types
import scannerpy.libscanner as bindings
import scanner.metadata_pb2 as metadata_types
import scanner.source_args_pb2 as source_types
import scanner.sampler_args_pb2 as sampler_types
import scanner.engine.rpc_pb2 as rpc_types
import scanner.engine.rpc_pb2_grpc as grpc_types
import scanner.types_pb2 as misc_types
from google.protobuf.descriptor import FieldDescriptor

class ProtobufGenerator:
    def __init__(self, cfg):
        self._mods = []
        for mod in [misc_types, rpc_types, grpc_types, metadata_types,
                    source_types, sampler_types, stdlib_types]:
            self.add_module(mod)

    def add_module(self, path):
        if isinstance(path, basestring):
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
    args_proto = getattr(protos, proto_name)()

    def analyze_proto(p):
        fields = {}
        for f in p.DESCRIPTOR.fields:
            child_fields = None
            if f.type == FieldDescriptor.TYPE_MESSAGE:
                child_fields = analyze_proto(f.message_type)

            fields[f.name] = {
                'type': f.type,
                'message': f.message_type,
                'repeated': f.label == FieldDescriptor.LABEL_REPEATED,
                'fields': child_fields,
            }
        return fields

    p = analyze_proto(args_proto)

    def serialize_obj(args_proto, p, obj):
        if isinstance(obj, dict):
            for k, v in obj.iteritems():
                if k not in p:
                    raise ScannerException(
                        'Protobuf does not have field {:s}'.format(k))
                desc = p[k]
                # If a message field
                def make_field(val):
                    if desc['type'] == FieldDescriptor.TYPE_MESSAGE:
                        # If a message field, we need to recursively invoke
                        # serialization
                        return serialize_obj(
                            desc['message'], desc['fields'], val)
                    else:
                        return val
                if p[k]['repeated']:
                    # If a repeated field, we need to set using slicing
                    data = []
                    for vi in v:
                        data.append(make_field(vi))
                    getattr(args_proto, k)[:] = data
                else:
                    # Just set the regular field
                    setattr(args_proto, k, make_field(v))
        else:
            raise ScannerException('Attempted to bind a non-dict type to a '
                                   'protobuf')

    serialize_obj(args_proto, p, obj)

    return args_proto.SerializeToString()
