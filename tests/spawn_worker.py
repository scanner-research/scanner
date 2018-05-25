from scannerpy import ProtobufGenerator, Config, start_worker
import time
import grpc
import sys

c = Config(None)

import scanner.metadata_pb2 as metadata_types
import scanner.engine.rpc_pb2 as rpc_types
import scanner.types_pb2 as misc_types
import scannerpy._python as bindings

con = Config(config_path='/tmp/config_test')
protobufs = ProtobufGenerator(con)

master_address = str(con.master_address) + ':' + str(con.master_port)
port = int(sys.argv[1])

params = bindings.default_machine_params()
mp = protobufs.MachineParameters()
mp.ParseFromString(params)
del mp.gpu_ids[:]
params = mp.SerializeToString()

start_worker(master_address, machine_params=params, config=con, block=True,
             port=port,
             watchdog=False)
