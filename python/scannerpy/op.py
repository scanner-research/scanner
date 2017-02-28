from common import *
import grpc

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
            return lambda: Op.input(self._db)
        elif name == 'Output':
            return lambda inputs: Op.output(self._db, inputs)

        has_op_args = self._db.protobufs.HasOpArgs()
        has_op_args.op_name = name

        try:
            result = self._db._master.HasOp(has_op_args)
        except grpc.RpcError as e:
            raise ScannerException(e)

        if not result.success:
            raise ScannerException('Op {} does not exist'.format(name))

        def make_op(**kwargs):
            inputs = kwargs.pop('inputs', [])
            device = kwargs.pop('device', DeviceType.CPU)
            args = kwargs.pop('args', None)
            return Op(self._db, name, inputs, device,
                             kwargs if args is None else args)
        return make_op


class Op:
    def __init__(self, db, name, inputs, device, args):
        self._db = db
        self._name = name
        self._inputs = inputs
        self._device = device
        self._args = args

    @classmethod
    def input(cls, db):
        # TODO(wcrichto): allow non-frame inputs
        return cls(db, "InputTable", [(None, ["frame", "frame_info"])],
                   DeviceType.CPU, {})

    @classmethod
    def output(cls, db, inputs):
        return cls(db, "OutputTable", inputs, DeviceType.CPU, {})

    def to_proto(self, indices):
        e = self._db.protobufs.Op()
        e.name = self._name

        for (in_eval, cols) in self._inputs:
            inp = e.inputs.add()
            idx = indices[in_eval] if in_eval is not None else -1
            inp.op_index = idx
            inp.columns.extend(cols)

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
