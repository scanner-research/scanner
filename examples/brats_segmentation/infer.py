from scannerpy import Database, DeviceType, Job, ColumnType
from scannerpy.stdlib import NetDescriptor, parsers, pipelines
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

script_dir = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = "/root/shared/u-net-brain-tumor/dataset/"
INPUT_DIR_BASE = DATASET_DIR + "input/"
OUTPUT_DIR_BASE = DATASET_DIR + "output/"
img_input_dir = INPUT_DIR_BASE + "Brats17_2013_0_1/"
img_output_dir = OUTPUT_DIR_BASE + "Brats17_2013_0_1/"

if not os.path.exists(img_output_dir):
    os.makedirs(img_output_dir)

with Database() as db:
    print('Populate the table ...')
    db.new_table('brats', ['img_input_dir', 'img_output_dir'], [[img_input_dir, img_output_dir],], force=True)
    input_table = db.table('brats')

    print('Brats segmentation ...')
    db.register_op('BratsSeg', ['img_input_dir', 'img_output_dir'], ['result'])
    db.register_python_kernel('BratsSeg', DeviceType.CPU,
                              script_dir + '/unet_kernel.py')
    img_input_dir, img_output_dir = input_table.as_op().range(0, 1)
    result = db.ops.BratsSeg(img_input_dir = img_input_dir,
                            img_output_dir = img_output_dir)
    job = Job(columns = [result], name = 'result')
    result_table = db.run(job, force=True)
