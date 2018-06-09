import scannerpy
import os
from datetime import datetime
import sys
import cv2
from scannerpy import Database, DeviceType, Job, FrameType

from PIL import Image

from reid import models
from reid.utils.serialization import load_checkpoint
from reid.feature_extraction import extract_cnn_feature
from reid.utils.data import transforms as T

from torch import nn

import torch


def init_model(model_path):
    # model = models.create('resnet152', num_features=1024, dropout=0, num_classes=2509)
    model = models.create('resnet50', num_features=128, num_classes=216, dropout=0)
    checkpoint = load_checkpoint(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(model).cuda()
    return model


@scannerpy.register_python_op(device_type=DeviceType.GPU)
class ExtractFeatureClass(scannerpy.Kernel):
    def __init__(self, config):
        self.model_path = config.args['model_path']
        self.width = config.args['width']
        self.height = config.args['height']
        self.model = init_model(self.model_path)
        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.test_transformer = T.Compose([
            T.RectScale(self.height, self.width),
            T.ToTensor(),
            self.normalizer,
        ])

    def execute(self, frame: FrameType) -> FrameType:
        trans_im = cv2.resize(frame, (self.width, self.height))

        image = Image.fromarray(trans_im)
        img = self.test_transformer(image)
        img_list = [img]
        imgs = torch.stack(img_list)

        t1 = datetime.now()

        img_feat = extract_cnn_feature(self.model, imgs)[0]

        t2 = datetime.now()

        # print("extract features in ", (t2 - t1).total_seconds() * 1000, " ms.")
        # print(img_feat)

        return trans_im


if len(sys.argv) <= 2:
    print('Usage: main.py <video_file> <open-reid model_path>')
    exit(1)

movie_path = sys.argv[1]

print('Extract features in video {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]

openreid_model_path = sys.argv[2]
print("OpenReId Model Path", openreid_model_path)

db = Database()

if not db.has_table(movie_name):
    print("Ingesting video into Scanner ...")
    db.ingest_videos([(movie_name, movie_path)], force=True)

input_table = db.table(movie_name)

sampler = db.streams.All
sampler_args = {}

if db.has_gpu():
    print("Using GPUs")
    device = DeviceType.GPU
    pipeline_instances = -1
else:
    print("Using CPUs")
    device = DeviceType.CPU
    pipeline_instances = 1

frame = db.sources.FrameColumn()

resized_frame_result = db.ops.ExtractFeatureClass(frame=frame,
                                                  model_path=openreid_model_path,
                                                  width=128, height=256,
                                                  device=DeviceType.GPU)

output = db.sinks.FrameColumn(columns={
    'resized_frame': resized_frame_result
})

job = Job(op_args={
    frame: input_table.column('frame'),
    output: 'example_python_op'
})

[table] = db.run(output=output, jobs=[job], force=True)

table.column('resized_frame').save_mp4('resized_frame')

print("Finished")
