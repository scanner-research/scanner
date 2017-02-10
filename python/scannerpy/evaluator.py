from common import *


class EvaluatorGenerator:
    """
    Creates Evaluator instances to define a computation.

    When a particular evaluator is requested from the generator, e.g.
    `db.evaluators.Histogram`, the generator does a dynamic lookup for the
    evaluator in a C++ registry.
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        if name == 'Input':
            return lambda: Evaluator.input(self._db)
        elif name == 'Output':
            return lambda inputs: Evaluator.output(self._db, inputs)

        if not self._db._bindings.has_evaluator(name):
            raise ScannerException('Evaluator {} does not exist'.format(name))

        def make_evaluator(**kwargs):
            inputs = kwargs.pop('inputs', [])
            device = kwargs.pop('device', DeviceType.CPU)
            args = kwargs.pop('args', None)
            return Evaluator(self._db, name, inputs, device,
                             kwargs if args is None else args)
        return make_evaluator


class Evaluator:
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

    def output_columns(self):
        # TODO
        pass

    def to_proto(self, indices):
        e = self._db._metadata_types.Evaluator()
        e.name = self._name

        for (in_eval, cols) in self._inputs:
            inp = e.inputs.add()
            idx = indices[in_eval] if in_eval is not None else -1
            inp.evaluator_index = idx
            inp.columns.extend(cols)

        e.device_type = DeviceType.to_proto(self._db, self._device)

        if isinstance(self._args, dict):
            # To convert an arguments dict, we search for a protobuf with the
            # name {Evaluator}Args (e.g. BlurArgs, HistogramArgs) in the
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
