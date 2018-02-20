from __future__ import absolute_import, division, print_function, unicode_literals
import grpc
import copy

from scannerpy.common import *
from scannerpy.protobuf_generator import python_to_proto

class OpColumn:
    def __init__(self, db, op, col, typ):
        self._db = db
        self._op = op
        self._col = col
        self._type = typ
        self._encode_options = None
        if self._type == self._db.protobufs.Video:
            self._encode_options = {'codec': 'default'}

    def sample(self):
        return self._db.ops.Sample(col=self)

    def space(self):
        return self._db.ops.Space(col=self)

    def slice(self):
        return self._db.ops.Slice(col=self)

    def unslice(self):
        return self._db.ops.Unslice(col=self)

    def compress(self, codec = 'video', **kwargs):
        self._assert_is_video()
        codecs = {'video': self.compress_video,
                  'default': self.compress_default,
                  'raw': self.lossless}
        if codec in codecs:
            return codecs[codec](self, **kwargs)
        else:
            raise ScannerException('Compression codec {} not currently '
                                   'supported. Available codecs are: {}.'
                                   .format(' '.join(codecs.keys())))

    def compress_video(self, quality = -1, bitrate = -1, keyframe_distance = -1):
        self._assert_is_video()
        encode_options = {
            'codec': 'h264',
            'quality': quality,
            'bitrate': bitrate,
            'keyframe_distance': keyframe_distance
        }
        return self._new_compressed_column(encode_options)

    def lossless(self):
        self._assert_is_video()
        encode_options = {'codec': 'raw'}
        return self._new_compressed_column(encode_options)

    def compress_default(self):
        self._assert_is_video()
        encode_options = {'codec': 'default'}
        return self._new_compressed_column(encode_options)

    def _assert_is_video(self):
        if self._type != self._db.protobufs.Video:
            raise ScannerException(
                'Compression only supported for columns of'
                'type "video". Column {} type is {}.'
                .format(self._col,
                        self.db.protobufs.ColumnType.Name(self._type)))

    def _new_compressed_column(self, encode_options):
        new_col = OpColumn(self._db, self._op, self._col, self._type)
        new_col._encode_options = encode_options
        return new_col


class OpGenerator:
    """
    Creates Op instances to define a computation.

    When a particular op is requested from the generator, e.g.
    `db.ops.Histogram`, the generator does a dynamic lookup for the
    op in a C++ registry.
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        if name == 'Input':
            return lambda: Op.input(self._db).outputs()
        elif name == 'FrameInput':
            return lambda: Op.frame_input(self._db).outputs()
        elif name == 'Output':
            def make_op(columns):
                op = Op.output(self._db, columns)
                return op
            return make_op

        # This will raise an exception if the op does not exist.
        op_info = self._db._get_op_info(name)

        def make_op(*args, **kwargs):
            inputs = []
            if op_info.variadic_inputs:
                inputs.extend(args)
            else:
                for c in op_info.input_columns:
                    val = kwargs.pop(c.name, None)
                    if val is None:
                        raise ScannerException('Op {} required column {} as input'
                                               .format(name, c.name))
                    inputs.append(val)
            device = kwargs.pop('device', DeviceType.CPU)
            batch = kwargs.pop('batch', -1)
            warmup = kwargs.pop('warmup', 0)
            stencil = kwargs.pop('stencil', [])
            args = kwargs.pop('args', None)
            op = Op(self._db, name, inputs, device, batch, warmup,
                    stencil, kwargs if args is None else args)
            return op.outputs()

        return make_op


class Op:
    def __init__(self, db, name, inputs, device, batch=-1, warmup=0,
                 stencil=[0], args={}):
        self._db = db
        self._name = name
        self._inputs = inputs
        self._device = device
        self._batch = batch
        self._warmup = warmup
        self._stencil = stencil
        self._args = args

        if (name == 'Input' or
            name == 'Space' or
            name == 'Sample' or
            name == 'Slice' or
            name == 'Unslice'):
            outputs = []
            for c in inputs:
                outputs.append(OpColumn(db, self, c._col, c._type))
        elif name == "OutputTable":
            columns = inputs
            self._output_names = [n for n, _ in columns.iteritems()]
            self._inputs = [c for _, c in columns.iteritems()]
            outputs = []
        else:
            cols = self._db._get_output_columns(self._name)
            outputs = [OpColumn(self._db, self, c.name, c.type) for c in cols]
        self._outputs = outputs

    @classmethod
    def input(cls, db):
        c = cls(db, "Input", [OpColumn(db, None, 'col', db.protobufs.Other)],
                DeviceType.CPU)
        return c

    @classmethod
    def frame_input(cls, db):
        c = cls(db, "Input", [OpColumn(db, None, 'col', db.protobufs.Video)],
                DeviceType.CPU)
        return c

    @classmethod
    def output(cls, db, inputs):
        return cls(db, "OutputTable", inputs, DeviceType.CPU)

    def inputs(self):
        return self._inputs

    def outputs(self):
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return tuple(self._outputs)

    def to_proto(self, indices):
        e = self._db.protobufs.Op()
        e.name = self._name
        e.device_type = DeviceType.to_proto(self._db.protobufs, self._device)
        e.stencil.extend(self._stencil)
        e.batch = self._batch
        e.warmup = self._warmup

        if e.name == "Input":
            inp = e.inputs.add()
            inp.column = self._inputs[0]._col
            inp.op_index = -1
        else:
            for i in self._inputs:
                inp = e.inputs.add()
                idx = indices[i._op] if i._op is not None else -1
                inp.op_index = idx
                inp.column = i._col

        if isinstance(self._args, dict):
            # To convert an arguments dict, we search for a protobuf with the
            # name {Op}Args (e.g. BlurArgs, HistogramArgs) in the
            # args.proto module, and fill that in with keys from the args dict.
            if len(self._args) > 0:
                proto_name = self._name + 'Args'
                e.kernel_args = python_to_proto(
                    self._db.protobufs, proto_name, self._args)
        else:
            # If arguments are a protobuf object, serialize it directly
            e.kernel_args = self._args.SerializeToString()

        return e
