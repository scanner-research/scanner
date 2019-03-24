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
    def __init__(self, config: KernelConfig, init_parameter = None):
        r"""

        Parameters
        ----------
        config
          Contains the configuration settings for this instance of the kernel.

        init_parameter
          An example init parameter. Any parameters defined in the __init__ method
          of a Kernel can be set when creating an instance of the corresponding operation.
          For example, an operation for this Kernel could be initialized like this:

          :code:`cl.ops.Kernel(init_parameter='test', ...)`
        """
        self.config = config

    def close(self):
        r"""Called when this Kernel instance will no longer be used.
        """
        pass

    def new_stream(self):
        r"""Runs after fetch_resources for each instance of this operation.

        Parameters added for this method by operations are considered
        `stream config parameters` (see :ref:`stream-config-parameters`).
        """
        pass

    def reset(self):
        r"""Called for stateful operations when the operation should reset its logical state.
        """
        pass

    def fetch_resources(self):
        r"""Runs once per Scanner worker to download resources for running this operation.
        """
        pass

    def setup_with_resources(self):
        r"""Runs after fetch_resources for each instance of this operation.

        This method is reponsible for handling any setup that requires resources to first be downloaded.
        """
        pass

    def execute(self, stream_parameter: bytes) -> bytes:
        r"""Runs the kernel on input elements and returns new output elements.

        Parameters
        ----------
        stream_parameter
          An example stream parameter. Must be annotated with a stream parameter type.
          See :ref:`stream-parameters`.

        Returns
        -------
        bytes
          The outputs for the operation.
        """

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
      elif msg_type == 'fetch_resources':
        kernel.fetch_resources()
        send_conn.send_bytes(b'')
      elif msg_type == 'setup_with_resources':
        kernel.setup_with_resources()
  except Exception as e:
    traceback.print_exc()
    raise
  finally:
    send_conn.close()
    recv_conn.close()
