
import grpc
import copy

from scannerpy.common import *
from scannerpy.op import OpColumn
from scannerpy.protobuf_generator import python_to_proto


class Sink:
    def __init__(self, db, name, inputs, sink_args={}):
        self._db = db
        self._name = name
        self._args = sink_args

        sink_info = self._db._get_sink_info(self._name)
        cols = sink_info.input_columns
        variadic_inputs = sink_info.variadic_inputs

        # TODO: Verify columns are the correct type here
        if name == 'FrameColumn' or name == 'Column':
            if 'columns' not in sink_args:
                raise ScannerException(
                    'Columns must be specified for Column Sink. For example, '
                    'db.sinks.Column(columns={\'column_name\': col}).')

            columns = sink_args['columns']
            self._output_names = [n for n, _ in columns.items()]
            self._inputs = [c for _, c in columns.items()]

            del sink_args['columns']
        else:
            self._output_names = []
            self._inputs = inputs

        if name == 'FrameColumn' or name == 'Column':
            # We insert the storage config to allow the ColumSink
            # to read from the database
            sc = self._db.config.config['storage']
            def check_and_add(key):
                if key in sc:
                    self._args[key] = sc[key]

            self._args['storage_type'] = sc['type']
            check_and_add('bucket')
            check_and_add('region')
            check_and_add('endpoint')

    def inputs(self):
        return self._inputs

    def to_proto(self, indices):
        e = self._db.protobufs.Op()
        e.name = self._name
        e.is_sink = True

        for i in self._inputs:
            inp = e.inputs.add()
            idx = indices[i._op] if i._op is not None else -1
            inp.op_index = idx
            inp.column = i._col

        if isinstance(self._args, dict):
            # To convert an arguments dict, we search for a protobuf with the
            # name {Name}SourceArgs (e.g. ColumnSourceArgs) and the name
            # {Name}EnumeratorArgs (e.g. ColumnEnumeratorArgs) in the
            # args.proto module, and fill that in with keys from the args dict.
            if len(self._args) > 0:
                sink_info = self._db._get_sink_info(self._name)
                if len(sink_info.protobuf_name) > 0:
                    proto_name = sink_info.protobuf_name
                    e.kernel_args = python_to_proto(
                        self._db.protobufs, proto_name, self._args)
                else:
                    e.kernel_args = self._args
        else:
            # If arguments are a protobuf object, serialize it directly
            e.kernel_args = self._args.SerializeToString()

        return e


class SinkGenerator:
    """
    Creates Sink instances to define a computation.

    When a particular Sink is requested from the generator, e.g.
    `db.sink.Column`, the generator does a dynamic lookup for the
    Sink in the servers registry.
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        # This will raise an exception if the source does not exist.
        sink_info = self._db._get_sink_info(name)

        def make_sink(*args, **kwargs):
            inputs = []
            if sink_info.variadic_inputs:
                inputs.extend(args)
            else:
                for c in sink_info.input_columns:
                    val = kwargs.pop(c.name, None)
                    if val is None:
                        raise ScannerException(
                            'sink {} required column {} as input'
                            .format(name, c.name))
                    inputs.append(val)

            sink_args = kwargs.pop('args', kwargs)
            sink = Sink(self._db, name,
                        inputs,
                        kwargs if args is None else sink_args)
            return sink

        return make_sink
