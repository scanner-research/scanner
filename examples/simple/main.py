import grpc

import sys
scanner_path = '/home/wcrichto/scanner'
sys.path.append(scanner_path + '/build')
sys.path.append(scanner_path + '/thirdparty/build/bin/storehouse/lib')

from storehousepy import StorageConfig
import scanner.metadata_pb2 as metadata
import scanner.engine.rpc_pb2 as rpc
import scanner.kernels.args_pb2 as kernel_args

import scanner_bindings

storage_config = StorageConfig.make_posix_config()

db_path = '/tmp/new_scanner_db'

scanner_bindings.create_database(storage_config, db_path)
scanner_bindings.ingest_videos(
    storage_config,
    db_path,
    ['meangirls'],
    ['/bigdata/wcrichto/videos/meanGirls_short.mp4'])

memory_config = metadata.MemoryPoolConfig()
memory_config.use_pool = False
db_params = scanner_bindings.make_database_parameters(
    storage_config, memory_config.SerializeToString(), db_path)

master_address = "localhost:5001"
master = scanner_bindings.start_master(db_params)
worker = scanner_bindings.start_worker(db_params, master_address)

job_params = rpc.JobParameters()
job_params.job_name = "test_job"

task_set = job_params.task_set
task = task_set.tasks.add()
task.output_table_name = "blurred_mean"
sample = task.samples.add()
sample.table_name = "meangirls"
sample.column_names.extend(["frame", "frame_info"])
sample.rows.extend(range(1000))

args = kernel_args.BlurArgs()
args.kernel_size = 3
args.sigma = 0.5

input = task_set.evaluators.add()
input.name = "InputTable"
input.device_type = metadata.CPU
input_input = input.inputs.add()
input_input.evaluator_index = -1
input_input.columns.extend(["frame", "frame_info"])

blur = task_set.evaluators.add()
blur.name = "Blur"
blur_input = blur.inputs.add()
blur_input.evaluator_index = 0
blur_input.columns.extend(["frame", "frame_info"])
blur.device_type = metadata.CPU
blur.kernel_args = args.SerializeToString()

output = task_set.evaluators.add()
output.name = "OutputTable"
output_input = output.inputs.add()
output_input.evaluator_index = 1
output_input.columns.extend(["frame"])

channel = grpc.insecure_channel(master_address)
stub = rpc.MasterStub(channel)
stub.NewJob(job_params)
