import grpc
import copy

from scannerpy.common import *
from scannerpy.op import OpColumn
from scannerpy.protobuf_generator import python_to_proto


class Source:
    def __init__(self, db, name, source_args={}):
        self._db = db
        self._name = name
        self._args = source_args

        source_info = self._db._get_source_info(self._name)
        cols = source_info.output_columns
        outputs = [OpColumn(self._db, self, c.name, c.type) for c in cols]
        self._outputs = outputs
        self._inputs = [OpColumn(self._db, None, c.name, c.type) for c in cols]

        # For the ColumnSource, we insert the storage config to allow the source
        # to read from the database
        if name == 'FrameColumn' or name == 'Column':
            sc = self._db.config.config['storage']

            def check_and_add(key):
                if key in sc:
                    self._args[key] = sc[key]

            self._args['storage_type'] = sc['type']
            check_and_add('bucket')
            check_and_add('region')
            check_and_add('endpoint')

            if 'load_sparsity_threshold' not in self._args:
                self._args['load_sparsity_threshold'] = 8

    def outputs(self):
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return tuple(self._outputs)

    def to_proto(self, indices):
        e = self._db.protobufs.Op()
        e.name = self._name
        e.is_source = True

        inp = e.inputs.add()
        inp.column = self._inputs[0]._col
        inp.op_index = -1

        if isinstance(self._args, dict):
            # To convert an arguments dict, we search for a protobuf with the
            # name {Name}SourceArgs (e.g. ColumnSourceArgs) and the name
            # {Name}EnumeratorArgs (e.g. ColumnEnumeratorArgs) in the
            # args.proto module, and fill that in with keys from the args dict.
            if len(self._args) > 0:
                source_info = self._db._get_source_info(self._name)
                if len(source_info.protobuf_name) > 0:
                    proto_name = source_info.protobuf_name
                    e.kernel_args = python_to_proto(self._db.protobufs,
                                                    proto_name, self._args)
                else:
                    e.kernel_args = self._args
        else:
            # If arguments are a protobuf object, serialize it directly
            e.kernel_args = self._args.SerializeToString()

        return e


class SourceGenerator:
    """
    Creates Source instances to define a computation.

    When a particular Source is requested from the generator, e.g.
    `db.source.Column`, the generator does a dynamic lookup for the
    Source in the servers registry.
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        # This will raise an exception if the source does not exist.
        source_info = self._db._get_source_info(name)

        def make_source(*args, **kwargs):
            source = Source(self._db, name, kwargs)
            return source.outputs()

        return make_source
