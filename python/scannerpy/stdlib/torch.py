from ..kernel import Kernel
from scannerpy import DeviceType
import sys


class TorchKernel(Kernel):
    def __init__(self, config):
        import torch

        self.config = config

        self.cpu_only = True
        visible_device_list = []
        for handle in config.devices:
            if int(handle.type) == DeviceType.GPU.value:
                visible_device_list.append(handle.id)
                self.cpu_only = False

        self.model = self.build_model()

        if not self.cpu_only:
            print('Using GPU: {}'.format(visible_device_list[0]))
            torch.cuda.set_device(visible_device_list[0])
            self.model = self.model.cuda()
        else:
            print('Using CPU')

        # Not sure if this is necessary? Haotian had it in his code
        self.model.eval()

    def images_to_tensor(self, images):
        import torch

        shape = images[0].shape
        images_tensor = torch.Tensor(len(images), shape[0], shape[1], shape[2])
        for i in range(len(images)):
            images_tensor[i] = images[i]
        return images_tensor

    def build_model(self):
        import torch

        sys.path.insert(0, os.path.split(self.config.args['model_def_path'])[0])
        kwargs = {
            'map_location': lambda storage, location: storage
        } if self.cpu_only else {}
        return torch.load(self.config.args['model_path'],
                          **kwargs)[self.config.args['model_key']]

    def close(self):
        del self.model

    def execute(self):
        raise NotImplementedError
