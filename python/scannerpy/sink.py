
import grpc
import copy

from scannerpy.common import *
from scannerpy.op import OpColumn, collect_per_stream_args, check_modules
from scannerpy.protobufs import python_to_proto, protobufs, analyze_proto


class Sink:
    def __init__(self, sc, name, inputs, job_args, sink_args={}):
        self._sc = sc
        self._name = name
        self._args = sink_args
        self._job_args = job_args

        sink_info = self._sc._get_sink_info(self._name)
        cols = sink_info.input_columns
        variadic_inputs = sink_info.variadic_inputs

        # TODO: Verify columns are the correct type here
        if name == 'FrameColumn' or name == 'Column':
            if 'columns' not in sink_args:
                raise ScannerException(
                    'Columns must be specified for Column Sink. For example, '
                    'sc.sinks.Column(columns={\'column_name\': col}).')

            columns = sink_args['columns']
            self._output_names = [n for n, _ in columns.items()]
            self._inputs = [c for _, c in columns.items()]

            del sink_args['columns']
        else:
            self._output_names = ['']
            self._inputs = inputs

        if name == 'FrameColumn' or name == 'Column':
            # We insert the storage config to allow the ColumSink
            # to read from the database
            sc = self._sc.config.config['storage']
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
        e = protobufs.Op()
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
                sink_info = self._sc._get_sink_info(self._name)
                if len(sink_info.protobuf_name) > 0:
                    proto_name = sink_info.protobuf_name
                    e.kernel_args = python_to_proto(proto_name, self._args)
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
    `sc.sink.Column`, the generator does a dynamic lookup for the
    Sink in the servers registry.
    """

    def __init__(self, sc):
        self._sc = sc

    def __getattr__(self, name):
        check_modules(self._sc)

        # Use Sequence as alias of Column
        if name == 'Sequence' or name == 'FrameSequence':
            name = name.replace('Sequence', 'Column')
            def make_sink(*args, **kwargs):
                column_name = 'frame' if 'Frame' in name else 'column'
                return Sink(self._sc, name, [], dict(columns={column_name: args[0]}))
            return make_sink
        else:
            # This will raise an exception if the source does not exist.
            sink_info = self._sc._get_sink_info(name)

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

                if name == 'Column' or name == 'FrameColumn':
                    job_args = [s.encode('utf-8') for s in kwargs.pop('table_name', None)]
                    kwargs.pop('column_name', None)
                else:
                    assert sink_info.stream_protobuf_name != ''
                    job_args = collect_per_stream_args(name, sink_info.stream_protobuf_name, kwargs)

                sink_args = kwargs.pop('args', kwargs)
                sink = Sink(self._sc, name,
                            inputs,
                            job_args,
                            kwargs if args is None else sink_args)
                return sink

            return make_sink
