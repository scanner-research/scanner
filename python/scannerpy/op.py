import grpc
import copy
import pickle
import types as pytypes
import uuid
import collections

from scannerpy.common import *
from scannerpy.protobufs import python_to_proto, protobufs, analyze_proto
from scannerpy import types as scannertypes
from typing import Dict, List, Union, Tuple, Optional, Sequence
from inspect import signature
from itertools import islice
from collections import OrderedDict
from functools import wraps


class SliceList(list):
    """List of per-stream arguments to each slice of an op."""

    pass


def collect_per_stream_args(name, protobuf_name, kwargs):
    stream_arg_names = list(analyze_proto(getattr(protobufs, protobuf_name)).keys())
    stream_args = {k: kwargs.pop(k) for k in stream_arg_names if k in kwargs}

    if len(stream_args) == 0:
        raise ScannerException(
            "Op `{}` received no per-stream arguments. Options: {}" \
            .format(name, ', '.join(stream_args)))

    N = [len(v) for v in stream_args.values() if isinstance(v, list)][0]

    job_args = [
        python_to_proto(protobuf_name, {
            k: v[i] if isinstance(v, list) else v
            for k, v in stream_args.items()
            if v is not None
        })
        for i in range(N)
    ]

    return job_args


class OpColumn:
    def __init__(self, sc, op, col, typ):
        self._sc = sc
        self._op = op
        self._col = col
        self._type = typ
        self._encode_options = None
        if self._type == protobufs.Video:
            self._encode_options = {'codec': 'default'}

    def compress(self, codec='video', **kwargs):
        self._assert_is_video()
        codecs = {
            'video': self.compress_video,
            'default': self.compress_default,
            'raw': self.lossless
        }
        if codec in codecs:
            return codecs[codec](self, **kwargs)
        else:
            raise ScannerException('Compression codec {} not currently '
                                   'supported. Available codecs are: {}.'
                                   .format(' '.join(list(codecs.keys()))))

    def compress_video(self, quality=-1, bitrate=-1, keyframe_distance=-1):
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
        if self._type != protobufs.Video:
            raise ScannerException('Compression only supported for sequences of'
                                   'type "video". Sequence {} type is {}.'.format(
                                       self._col,
                                       protobufs.ColumnType.Name(
                                           self._type)))

    def _new_compressed_column(self, encode_options):
        new_col = OpColumn(self._sc, self._op, self._col, self._type)
        new_col._encode_options = encode_options
        return new_col


MODULE_REGISTRY = set()


def register_module(so_path: str, proto_path: str = None):
    MODULE_REGISTRY.add((so_path, proto_path))


def check_modules(sc):
    unregistered_modules = MODULE_REGISTRY - sc._modules
    for module in unregistered_modules:
        sc.load_op(*module)


PYTHON_OP_REGISTRY = {}


class OpGenerator:
    """
    Creates Op instances to define a computation.

    When a particular op is requested from the generator, e.g.
    `sc.ops.Histogram`, the generator does a dynamic lookup for the
    op in a C++ registry.
    """

    def __init__(self, sc):
        self._sc = sc

    def __getattr__(self, name):
        check_modules(self._sc)

        stream_params = []

        orig_name = name

        # Check python registry for Op
        if name in PYTHON_OP_REGISTRY:
            py_op_info = PYTHON_OP_REGISTRY[name]
            stream_params = py_op_info['stream_params']
            # If Op has not been registered yet, register it
            pseudo_name = name + ':' + py_op_info['registration_id']
            name = pseudo_name
            if not name in self._sc._python_ops:
                devices = []
                if py_op_info['device_type']:
                    devices.append(py_op_info['device_type'])
                if py_op_info['device_sets']:
                    for d in py_op_info['device_sets']:
                        devices.append(d[0])

                self._sc.register_op(
                    pseudo_name, py_op_info['input_columns'],
                    py_op_info['output_columns'], py_op_info['variadic_inputs'],
                    py_op_info['stencil'], py_op_info['unbounded_state'],
                    py_op_info['bounded_state'], py_op_info['proto_path'])
                for device in devices:
                    self._sc.register_python_kernel(pseudo_name, device,
                                                    py_op_info['kernel'],
                                                    py_op_info['batch'])

        # This will raise an exception if the op does not exist.
        op_info = self._sc._get_op_info(name)

        if (not orig_name in PYTHON_OP_REGISTRY and
            len(op_info.stream_protobuf_name) > 0):
            stream_proto = getattr(protobufs, op_info.stream_protobuf_name)
            for proto_field_name, _ in analyze_proto(stream_proto).items():
                stream_params.append(proto_field_name)

        def make_op(*args, **kwargs):
            inputs = []
            if op_info.variadic_inputs:
                inputs.extend(args)
            else:
                for c in op_info.input_columns:
                    val = kwargs.pop(c.name, None)
                    if val is None:
                        raise ScannerException(
                            'Op {} required sequence {} as input'.format(
                                orig_name, c.name))
                    inputs.append(val)

            device = kwargs.pop('device', DeviceType.CPU)
            batch = kwargs.pop('batch', -1)
            bounded_state = kwargs.pop('bounded_state', -1)
            stencil = kwargs.pop('stencil', [])
            extra = kwargs.pop('extra', None)
            args = kwargs.pop('args', None)

            if len(stream_params) > 0:
                stream_args = [(k, kwargs.pop(k, None)) for k in stream_params
                               if k in kwargs]
                if len(stream_args) == 0:
                    raise ScannerException(
                           "No arguments provided to op `{}` for stream parameters."
                            .format(orig_name))
                for e in stream_args:
                    if not isinstance(e[1], list):
                        raise ScannerException(
                            "The argument `{}` to op `{}` is a stream config argument and must be a list."
                            .format(e[0], orig_name))
                example_list = stream_args[0][1]
                N = len(example_list)
                if not isinstance(example_list[0], SliceList):
                    stream_args = [
                        (k, [SliceList([x]) for x in arg]) for (k, arg) in stream_args]
                M = len(stream_args[0][1][0])

                if orig_name in PYTHON_OP_REGISTRY:
                    stream_args = [
                        SliceList([pickle.dumps({k: v[i][j] for k, v in stream_args})
                                   for j in range(M)])
                        for i in range(N)
                    ]
                else:
                    stream_args = [
                        SliceList([python_to_proto(op_info.stream_protobuf_name,
                                                   {k: v[i][j] for k, v in stream_args})
                                  for j in range(M)])
                        for i in range(N)
                    ]
            else:
                stream_args = None

            op = Op(self._sc, name, inputs, device, batch, bounded_state,
                    stencil, kwargs if args is None else args, extra, stream_args)
            return op.outputs()

        return make_op


class Op:
    def __init__(self,
                 sc,
                 name,
                 inputs,
                 device,
                 batch=-1,
                 warmup=-1,
                 stencil=[0],
                 args={},
                 extra=None,
                 stream_args=None):
        self._sc = sc
        self._name = name
        self._inputs = inputs
        self._device = device
        self._batch = batch
        self._warmup = warmup
        self._stencil = stencil
        self._args = args
        self._extra = extra
        self._job_args = stream_args

        if (name == 'Space' or name == 'Sample' or name == 'Slice'
                or name == 'Unslice'):
            outputs = []
            for c in inputs:
                outputs.append(OpColumn(sc, self, c._col, c._type))
        else:
            cols = self._sc._get_output_columns(self._name)
            outputs = [OpColumn(self._sc, self, c.name, c.type) for c in cols]
        self._outputs = outputs

    def inputs(self):
        return self._inputs

    def outputs(self):
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return tuple(self._outputs)

    def to_proto(self, indices):
        e = protobufs.Op()
        e.name = self._name
        e.device_type = DeviceType.to_proto(protobufs, self._device)
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
            if self._name in self._sc._python_ops:
                e.kernel_args = pickle.dumps(self._args)
            elif len(self._args) > 0:
                # To convert an arguments dict, we search for a protobuf with the
                # name {Op}Args (e.g. BlurArgs, HistogramArgs) in the
                # args.proto module, and fill that in with keys from the args dict.
                op_info = self._sc._get_op_info(self._name)
                if len(op_info.protobuf_name) > 0:
                    proto_name = op_info.protobuf_name
                    e.kernel_args = python_to_proto(proto_name, self._args)
                else:
                    e.kernel_args = self._args
        else:
            # If arguments are a protobuf object, serialize it directly
            e.kernel_args = self._args.SerializeToString()

        return e


def register_python_op(name: str = None,
                       stencil: List[int] = None,
                       unbounded_state: bool = False,
                       bounded_state: int = None,
                       device_type: DeviceType = None,
                       device_sets: List[Tuple[DeviceType, int]] = None,
                       batch: int = 1,
                       proto_path: str = None):
    r"""Class or function decorator which registers a new Op and Kernel with the
    Scanner master.

    Parameters
    ----------
    name
      Optional name for the Op. By default, it will be inferred as the name of the
      decorated class/kernel.

    stencil
      Specifies the default stencil to use for the Op. If none, indicates
      that the the Op does not have the ability to stencil. A stencil of
      [0] should be specified if the Op can stencil but should not by
      default.

    unbounded_state
      If true, indicates that the Op needs to see all previous elements
      of its input sequences before it can compute a given element. For
      example, to compute output element at index 100, the Op must have
      already produced elements 0-99. This option is mutually exclusive
      with `bounded_state`.

    bounded_state
      If true, indicates that the Op needs to see all previous elements
      of its input sequences before it can compute a given element. For
      example, to compute output element at index 100, the Op must have
      already produced elements 0-99. This option is mutually exclusive
      with `bounded_state`.

    device_type

    device_sets

    batch

    proto_path
      Optional path to the proto file that describes the configuration
      arguments to this Op.
    """
    def dec(fn_or_class):
        is_fn = False
        if isinstance(fn_or_class, pytypes.FunctionType) or isinstance(
                fn_or_class, pytypes.BuiltinFunctionType):
            is_fn = True

        if name is None:
            # Infer name from fn_or_class name
            kname = fn_or_class.__name__
        else:
            kname = name

        can_stencil = stencil is not None
        can_batch = batch > 1

        # Get execute function to determine input and output types
        if is_fn:
            exec_fn = fn_or_class
        else:
            exec_fn = getattr(fn_or_class, "execute", None)
            if not callable(exec_fn):
                raise ScannerException(
                    ('Attempted to register Python Op with name {:s}, but that '
                     'provided class has no "execute" method.').format(kname))

        input_columns = []
        has_variadic_inputs = False
        sig = signature(exec_fn)

        fn_params = sig.parameters
        if is_fn:
            # If this is a fn kernel, then the first argument should be `config`
            fn_params = OrderedDict(islice(fn_params.items(), 1, None))
        else:
            # If this is a class kernel, then first argument should be self
            fn_params = OrderedDict(islice(fn_params.items(), 1, None))

        def parse_annotation_to_column_type(typ, is_input=False):
            if can_batch:
                # If the op can batch, then we expect the types to be
                # Sequence[T], where T = {bytes, FrameType}
                if (not getattr(typ, '__origin__', None) or
                    (typ.__origin__ != Sequence and
                     typ.__origin__ != collections.abc.Sequence)):
                    raise ScannerException(
                        ('A batched Op must specify a "Sequence" type '
                         'annotation for each input and output.'))
                typ = typ.__args__[0]

            if is_input and can_stencil:
                # If the op can stencil, then we expect the input types to be
                # Sequence[T], where T = {bytes, FrameType}
                if (not getattr(typ, '__origin__', None) or
                    (typ.__origin__ != Sequence and
                     typ.__origin__ != collections.abc.Sequence)):
                    raise ScannerException(
                        ('A stenciled Op must specify a "Sequence" type '
                         'annotation for each input. If the Op both stencils '
                         'and batches, then it should have the type '
                         '"Sequence[Sequence[T]], where T = {bytes, FrameType}.'
                         ))
                typ = typ.__args__[0]

            if typ == scannertypes.FrameType:
                column_type = ColumnType.Video
            elif typ == bytes:
                column_type = ColumnType.Blob
            else:
                # For now, all non-FrameType types are equivalent to bytes.
                column_type = ColumnType.Blob
            return column_type, typ

        # Analyze exec_fn parameters to determine the input types
        for param_name, param in fn_params.items():
            # We only allow keyword arguments and *args.
            # There is no support currently for positional or **kwargs
            kind = param.kind
            if (kind == param.POSITIONAL_ONLY or kind == param.VAR_KEYWORD):
                raise ScannerException(
                    ('Positional arguments and **kwargs are currently not '
                     'supported for the "execute" method of kernels'))

            if kind == param.VAR_POSITIONAL:
                # This means we have variadic inputs
                has_variadic_inputs = True
                if len(fn_params) > 1:
                    raise ScannerException(
                        ('Variadic positional inputs (*args) are not supported '
                         'when used with other inputs.'))
                break

            if param.annotation == param.empty:
                raise ScannerException(
                    ('No type annotation specified for input {:s}. Must '
                     'specify an annotation of "bytes" or "FrameType".')
                    .format(param_name))

            typ = param.annotation
            column_type, typ = parse_annotation_to_column_type(typ, is_input=True)
            type_info = scannertypes.get_type_info(typ)
            input_columns.append((param_name, column_type, type_info))

        output_columns = []
        # Analyze exec_fn return type to determine output types
        typ = sig.return_annotation
        if typ == sig.empty:
            raise ScannerException(
                ('Return annotation must be specified for "execute" method.'))

        return_is_tuple = True
        origin_type = getattr(typ, '__origin__', None)
        if origin_type == Tuple or origin_type == tuple:
            if getattr(typ, '__tuple_params__', None):
                # Python 3.5
                use_ellipsis = typ.__tuple_use_ellipsis__
                tuple_params = typ.__tuple_params__
            elif getattr(typ, '__args__', None):
                # Python 3.6+
                use_ellipsis = typ.__args__[-1] is Ellipsis
                tuple_params = typ.__args__[:-1 if use_ellipsis else None]
            else:
                raise ScannerException('This should not happen...')
        else:
            use_ellipsis = False
            return_is_tuple = False
            tuple_params = [typ]

        if use_ellipsis:
            raise ScannerException(
                ('Ellipsis tuples not supported for return type.'))

        # Parse the return types into Scanner column types
        for i, typ in enumerate(tuple_params):
            column_type, typ = parse_annotation_to_column_type(typ)
            type_info = scannertypes.get_type_info(typ)
            output_columns.append(('ret{:d}'.format(i), column_type, type_info))

        if kname in PYTHON_OP_REGISTRY:
            raise ScannerException(
                'Attempted to register Op with name {:s} twice'.format(kname))

        def parse_params(in_cols):
            args = {}
            for (param_name, _1, type_info), cs in zip(input_columns, in_cols):
                if can_batch ^ can_stencil:
                    args[param_name] = [type_info.deserialize(c) for c in cs]
                elif can_batch and can_stencil:
                    args[param_name] = [[type_info.deserialize(c) for c in c2] for c2 in cs]
                else:
                    args[param_name] = type_info.deserialize(cs)
            return args

        def parse_ret(r):
            columns = r if return_is_tuple else (r, )
            outputs = []
            for (_1, _2, type_info), column in zip(output_columns, columns):
                if can_batch:
                    outputs.append([
                        type_info.serialize(element)
                        for element in column
                    ])
                else:
                    outputs.append(
                        type_info.serialize(column))
            return tuple(outputs)

        # Wrap exec_fn to destructure input and outputs to proper python inputs
        if is_fn:
            if has_variadic_inputs:

                @wraps(fn_or_class)
                def wrapper_exec(config, in_cols):
                    return parse_ret(exec_fn(config, *in_cols))
            else:

                @wraps(fn_or_class)
                def wrapper_exec(config, in_cols):
                    args = parse_params(in_cols)
                    return parse_ret(exec_fn(config, **args))

            wrapped_fn_or_class = wrapper_exec
        else:
            wrapped_fn_or_class = type(fn_or_class.__name__ + 'Kernel', fn_or_class.__bases__,
                                       dict(fn_or_class.__dict__))
            if has_variadic_inputs:

                def execute(self, in_cols):
                    return parse_ret(exec_fn(self, *in_cols))
            else:

                def execute(self, in_cols):
                    args = parse_params(in_cols)
                    return parse_ret(exec_fn(self, **args))
            wrapped_fn_or_class.execute = execute

        dtype = device_type
        if device_type is None and device_sets is None:
            dtype = DeviceType.CPU

        if device_type is not None and device_sets is not None:
            raise ScannerException(
                'Must only specify one of "device_type" or "device_sets" for python Op.')

        if not is_fn:
            init_params = list(signature(getattr(fn_or_class, "__init__")).parameters.keys())[1:]
            if init_params[0] != 'config':
                raise ScannerException("__init__ first argument (after self) must be `config`")
            kernel_params = init_params[1:]

            stream_params = list(signature(getattr(fn_or_class, "new_stream")).parameters.keys())[1:]
        else:
            kernel_params = []
            stream_params = []

        PYTHON_OP_REGISTRY[kname] = {
            'input_columns': input_columns,
            'output_columns': output_columns,
            'variadic_inputs': has_variadic_inputs,
            'stencil': stencil,
            'unbounded_state': unbounded_state,
            'bounded_state': bounded_state,
            'kernel': wrapped_fn_or_class,
            'device_type': dtype,
            'device_sets': device_sets,
            'batch': batch,
            'proto_path': proto_path,
            'registration_id': uuid.uuid4().hex,
            'kernel_params': kernel_params,
            'stream_params': stream_params
        }
        return fn_or_class

    return dec
