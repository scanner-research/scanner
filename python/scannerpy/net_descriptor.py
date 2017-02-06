from common import *
import toml


class NetDescriptor:
    def __init__(self, db):
        self._descriptor = db.protobufs.NetDescriptor()

    def _val(self, dct, key, default):
        if key in dct:
            return dct[key]
        else:
            return default

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
