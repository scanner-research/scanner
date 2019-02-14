import pickle
import traceback

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

    def close(self):
        pass

    def new_stream(self):
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

  # Close parent connections
  p_conn1.close()
  p_conn2.close()
  try:
    kernel_config = KernelConfig(cloudpickle.loads(n['config']))
    kernel = cloudpickle.loads(n['kernel_code'])(kernel_config, **kernel_config.args)
    while True:
      try:
        data = recv_conn.recv_bytes()
        msg_type, data = cloudpickle.loads(data)
      except EOFError as e:
        break
      if msg_type == 'reset':
        kernel.reset()
      elif msg_type == 'new_stream':
        kernel.new_stream(**(data if data is not None else {}))
      elif msg_type == 'execute':
        result = kernel.execute(data)
        send_conn.send_bytes(cloudpickle.dumps(result))
  except Exception as e:
    traceback.print_exc()
    raise
  finally:
    send_conn.close()
    recv_conn.close()
