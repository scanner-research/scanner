
import toml

from scannerpy.common import *

class NetDescriptor(object):
    def __init__(self, db):
        self._descriptor = db.protobufs.NetDescriptor()
        self._descriptor.input_width = -1
        self._descriptor.input_height = -1
        self._descriptor.pad_mod = -1

    def _val(self, dct, key, default):
        if key in dct:
            return dct[key]
        else:
            return default

    @property
    def model_path(self):
        return self._descriptor.model_path

    @model_path.setter
    def model_path(self, value):
        self._descriptor.model_path = value

    @property
    def model_weights_path(self):
        return self._descriptor.model_weights_path

    @model_weights_path.setter
    def model_weights_path(self, value):
        self._descriptor.model_weights_path = value

    @property
    def input_layer_names(self):
        return self._descriptor.input_layer_names[:]

    @input_layer_names.setter
    def input_layer_names(self, value):
        del self._descriptor.input_layer_names[:]
        self._descriptor.input_layer_names.extend(value)

    @property
    def output_layer_names(self):
        return self._descriptor.output_layer_names[:]

    @output_layer_names.setter
    def output_layer_names(self, value):
        del self._descriptor.output_layer_names[:]
        self._descriptor.output_layer_names.extend(value)

    @property
    def input_width(self):
        return self._descriptor.input_width

    @input_width.setter
    def input_width(self, value):
        self._descriptor.input_width = value

    @property
    def input_height(self):
        return self._descriptor.input_height

    @input_width.setter
    def input_height(self, value):
        self._descriptor.input_height = value

    @property
    def normalize(self):
        return self._descriptor.normalize

    @normalize.setter
    def normalize(self, value):
        self._descriptor.normalize = value

    @property
    def preserve_aspect_ratio(self):
        return self._descriptor.preserve_aspect_ratio

    @preserve_aspect_ratio.setter
    def normalize(self, value):
        self._descriptor.preserve_aspect_ratio = value

    @property
    def transpose(self):
        return self._descriptor.transpose

    @transpose.setter
    def transpose(self, value):
        self._descriptor.transpose = value

    @property
    def pad_mod(self):
        return self._descriptor.pad_mod

    @pad_mod.setter
    def pad_mod(self, value):
        self._descriptor.pad_mod = value

    @property
    def uses_python(self):
        return self._descriptor.uses_python

    @uses_python.setter
    def uses_python(self, value):
        self._descriptor.uses_python = value

    @property
    def mean_colors(self):
        return self._descriptor.mean_colors

    @uses_python.setter
    def mean_colors(self, value):
        del self._descriptor.mean_colors[:]
        self._descriptor.mean_colors.extend(value)

    @classmethod
    def from_file(cls, db, path):
        self = cls(db)
        with open(path) as f:
            args = toml.loads(f.read())

        d = self._descriptor
        net = args['net']
        d.model_path = net['model']
        d.model_weights_path = net['weights']
        d.input_layer_names.extend(net['input_layers'])
        d.output_layer_names.extend(net['output_layers'])
        d.input_width = self._val(net, 'input_width', -1)
        d.input_height = self._val(net, 'input_height', -1)
        d.normalize = self._val(net, 'normalize', False)
        d.preserve_aspect_ratio = self._val(net, 'preserve_aspect_ratio', False)
        d.transpose = self._val(net, 'tranpose', False)
        d.pad_mod = self._val(net, 'pad_mod', -1)
        d.uses_python = self._val(net, 'uses_python', False)

        mean = args['mean-image']
        if 'colors' in mean:
            order = net['input']['channel_ordering']
            for color in order:
                d.mean_colors.append(mean['colors'][color])
        elif 'image' in mean:
            d.mean_width = mean['width']
            d.mean_height = mean['height']
            # TODO: load mean binaryproto
            raise ScannerException('TODO')

        return self

    def as_proto(self):
        return self._descriptor
