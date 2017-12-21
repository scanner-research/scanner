from __future__ import absolute_import, division, print_function, unicode_literals
import os.path
import imp
import sys

from scannerpy.common import *

import scannerpy.libscanner as bindings
import scanner.metadata_pb2 as metadata_types
import scanner.engine.rpc_pb2 as rpc_types
import scanner.engine.rpc_pb2_grpc as grpc_types
import scanner.types_pb2 as misc_types
import scanner.stdlib.stdlib_pb2 as stdlib_types

class ProtobufGenerator:
    def __init__(self, cfg):
        self._mods = []

        for mod in [misc_types, rpc_types, grpc_types, metadata_types,
                    stdlib_types]:
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
