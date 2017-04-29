from common import *
import grpc
import copy

class OpColumn:
    def __init__(self, db, op, col, typ):
        self._db = db
        self._op = op
        self._col = col
        self._type = typ
        self._encode_options = None
        if self._type == self._db.protobufs.Video:
            self._encode_options = {'codec': 'default'}

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

    def compress_video(self, quality = -1, bitrate = -1):
        self._assert_is_video()
        encode_options = {
            'codec': 'h264',
            'quality': quality,
            'bitrate': bitrate
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
            return lambda inputs, generator, collection: Op.input(self._db, inputs, generator, collection)
        elif name == 'Output':
            return lambda inputs: Op.output(self._db, inputs)

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
            args = kwargs.pop('args', None)
            op = Op(self._db, name, inputs, device,
                    kwargs if args is None else args)
            return op.outputs()

        return make_op


class Op:
    def __init__(self, db, name, inputs, device, args):
        self._db = db
        self._name = name
        self._inputs = inputs
        self._device = device
        self._args = args
        self._task = None

    @classmethod
    def input(cls, db, inputs, generator, collection):
        c = cls(db, "InputTable", inputs, DeviceType.CPU, {})
        c._generator = generator
        c._collection = collection
        return c

    @classmethod
    def output(cls, db, inputs):
        return cls(db, "OutputTable", inputs, DeviceType.CPU, {})

    def inputs(self):
        return self._inputs

    def outputs(self):
        if self._name == "InputTable":
            cols = [OpColumn(self._db, self, c.name(), c.type())
                    for c in self._inputs][1:]
        else:
            cols = self._db._get_output_columns(self._name)
            cols = [OpColumn(self._db, self, c.name, c.type) for c in cols]
        if len(cols) == 1:
            return cols[0]
        else:
            return tuple(cols)


    def to_proto(self, indices):
        e = self._db.protobufs.Op()
        e.name = self._name
        e.batch = -1

        if e.name == "InputTable":
            inp = e.inputs.add()
            inp.columns.extend([c.name() for c in self._inputs])
            inp.op_index = 0
        else:
            for i in self._inputs:
                inp = e.inputs.add()
                idx = indices[i._op] if i._op is not None else -1
                inp.op_index = idx
                inp.columns.append(i._col)

        e.device_type = DeviceType.to_proto(self._db, self._device)

        if isinstance(self._args, dict):
            # To convert an arguments dict, we search for a protobuf with the
            # name {Op}Args (e.g. BlurArgs, HistogramArgs) in the
            # args.proto module, and fill that in with keys from the args dict.
            if len(self._args) > 0:
                proto_name = self._name + 'Args'
                args_proto = getattr(self._db.protobufs, proto_name)()
                for k, v in self._args.iteritems():
                    try:
                        setattr(args_proto, k, v)
                    except AttributeError:
                        # If the attribute is a nested proto, we can't assign
                        # directly, so copy from the value.
                        getattr(args_proto, k).CopyFrom(v)
                    e.kernel_args = args_proto.SerializeToString()
        else:
            # If arguments are a protobuf object, serialize it directly
            e.kernel_args = self._args.SerializeToString()

        return e
