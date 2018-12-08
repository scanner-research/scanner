import pickle


class KernelConfig(object):
    def __init__(self, config):
        self.devices = config.devices
        self.input_columns = config.input_columns
        self.input_column_types = config.input_column_types
        self.output_columns = config.output_columns
        self.output_column_types = config.output_column_types
        self.args = pickle.loads(config.args()) if config.args() != b'' else None
        self.node_id = config.node_id


class Kernel(object):
    def __init__(self, config):
        self.config = config
        self.protobufs = config.protobufs

    def close(self):
        pass

    def new_stream(self, args):
        pass

    def reset(self):
        pass

    def execute(self, input_columns):
        raise NotImplementedError


def python_kernel_fn(n, recv_conn, send_conn, p_conn1, p_conn2):
  import pickle
  import cloudpickle
  import traceback
  import os
  from scannerpy import Config, DeviceType, DeviceHandle, KernelConfig
  from scannerpy.protobuf_generator import ProtobufGenerator

  # Close parent connections
  p_conn1.close()
  p_conn2.close()
  try:
    user_config = pickle.loads(n['user_config_str'])
    protobufs = ProtobufGenerator(user_config)
    kernel_config = KernelConfig(cloudpickle.loads(n['config']))
    kernel_config.protobufs = protobufs
    kernel = cloudpickle.loads(n['kernel_code'])(kernel_config)
    while True:
      try:
        data = recv_conn.recv_bytes()
        msg_type, data = cloudpickle.loads(data)
      except EOFError as e:
        break
      if msg_type == 'reset':
        kernel.reset()
      elif msg_type == 'new_stream':
        kernel.new_stream(data)
      elif msg_type == 'execute':
        result = kernel.execute(data)
        send_conn.send_bytes(cloudpickle.dumps(result))
  except Exception as e:
    print(e)
    raise
  finally:
    send_conn.close()
    recv_conn.close()
