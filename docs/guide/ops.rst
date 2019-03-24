.. _ops:

Operations
==========

Overview
--------
Processing data in Scanner occurs primarily through *operations* (*ops*). Operations are built-in or user defined functionality that transform streams of input elements to streams of output elements. For example, this :code:`Resize` operation transforms input frames by resizing them to a target width and height:

.. code-block:: python

   import scannerpy as sp
   import scannertools.imgproc

   cl = sp.Client()
   ...
   cl.ops.Resize(frame=input_frames, width=[640], height=[480])

Follow the rest of this guide to understand how to write your own operations.

Defining an Operation in Python
-------------------------------
Scanner supports defining operations in Python using the following syntax:

.. code-block:: python

   import scannerpy as sp
                
   @sp.register_python_op()
   class PersonDetectorDNN(sp.Kernel):
       # Init runs once when the class instance is initialized
       def __init__(self, config, model_url):
           self._model_url = model_url
           self._local_model_path = '/tmp/model'

       def fetch_resources(self):
           download_model(self._model_url, self._local_model_path)

       def setup_with_resources(self):
           with open(self._local_model_path, 'rb') as f:
               # load and setup the model
               self.model = ...

       def new_stream(self, threshold):
           self._threshold = threshold
    
       def execute(self, frame: sp.FrameType) -> BoundingBoxList:
           person_boxes, confidences = self.model.execute(frame)
           return [box
                   for box, conf in zip(person_boxes, confidences)
                   if conf > self._threshold]

We now have an operation defined named :code:`PersonDetectorDNN` which can be instantiated using a client object named :code:`sc` like this: :code:`sc.ops.PersonDetectorDNN(frame=..., model_url=..., threshold=[...])`. The rest of this guide will explain the above piece of code in more detail. First, let's start with the way parameters to the operation.

.. _declaring-parameters:

Declaring Parameters
--------------------
Parameters to operations can be declared at one of three locations:

- :py:meth:`~scannerpy.kernel.Kernel.__init__`: the constructor for the operation.
- :py:meth:`~scannerpy.kernel.Kernel.new_stream`: called whenever the operation starts processing a new stream.
- :py:meth:`~scannerpy.kernel.Kernel.execute`: called with input stream elements to produce transformed output elements.

The frequency at which each of these functions is invoked is called its *rate*. 

Init Parameters
~~~~~~~~~~~~~~~
Init parameters are specified once and given to the constructor of an operation, and then never changed. For example, the :code:`model_url` parameter to the :code:`PersonDetectorDNN` operation is a init parameter as it is specified in the :py:meth:`~scannerpy.kernel.Kernel.__init__` method to the class.

.. _stream-config-parameters:

Stream Config Parameters
~~~~~~~~~~~~~~~~~~~~~~~
Stream config parameters are provided to an operation before processing elements from a stream in a batch of streams (see :ref:`batch-processing`). For example, the :code:`threshold` parameter specified in the :py:meth:`~scannerpy.kernel.Kernel.new_stream` method of the :code:`PersonDetectorDNN` operation is a stream config parameter. For each stream in the batch of streams being processed, Scanner expects a separate :code:`threshold` parameter to be provided. This is done by passing a list like so: :code:`PersonDetectorDNN(threshold=[0.5, 0.6, ..., 0.2], ...)`.

.. _stream-parameters:

Stream Parameters
~~~~~~~~~~~~~~~~~
Stream parameters are parameters that bind to streams and are processed element at a time by the operation. For the :code:`PersonDetectorDNN` operation, the :code:`frame` parameter specified in the :py:meth:`~scannerpy.kernel.Kernel.execute` method is a stream parameter. Stream parameters must be annotated with a type so that Scanner understands how to serialize and deserialize the data. The following types are supported:

- :py:class:`~scannerpy.types.FrameType`: a built-in type that represents frames from a video. In Python operations, parameters of this type are represented with numpy arrays.
- :py:class:`~bytes`: these parameters represent blobs of binary data.
- User-defined types: Scanner supports registering custom types for stream parameters. See the next section to find out how to do that!

Stream Parameter Types
~~~~~~~~~~~~~~~~~~~~~~
Stream parameter types tell Scanner how to serialize and deserialize the elements in streams and the inputs/outputs to operations. One can define their own custom stream parameter type in Python by calling :py:func:`scannerpy.types.register_type`. For example, to register a new type for numpy float32 arrays, we can write the following code:

.. code-block:: python

   import scannerpy as sp
   import scannerpy.types
   import numpy as np

   @sp.types.register_type
   class NumpyArrayFloat32:
       def serialize(array):
           return array.tobytes()
    
       def deserialize(data_buffer):
           return np.fromstring(data_buffer, dtype=np.float32)

A custom type implements a :code:`serialize` method, which takes an instance of the type and converts it to a byte buffer, and a :code:`deserialize` method, which takes a byte buffer produced by calling :code:`serialize` and converts it back into an instance of the type.

Fetching Resources
------------------
Some operations require external resources to be download or fetched before they can start processing data. In the case of the :code:`PersonDetectorDNN` operation, it requires the model weights for its deep neural network. To download external resources, operations should implement the following two methods:

- :py:meth:`scannerpy.kernel.Kernel.fetch_resources`: To avoid redownloading resources that are shared across instances of an operation, this method is called once per machine to download resources.
- :py:meth:`scannerpy.kernel.Kernel.setup_with_resources`: Once :py:meth:`~scannerpy.kernel.Kernel.fetch_resources` has been executed, this method is called for every instance of that operation after :code:`__init__`.

Operation Properties
--------------------
Scanner operations can be annotated with several different properties to change their functionality.

Device Sets
~~~~~~~~~~~
By default, Scanner will assume operations only use the CPU when processing data. If an operation utilizes the GPU when processing elements, it can declare that it requires that device type during op declaration:

.. code-block:: python

   @sp.register_python_op(device_sets=[(DeviceType.GPU, 1)])
   class GpuOp():
       def __init__(self, config):
           pass

       ...

Batch
~~~~~
Many operations benefit from being able to process a *batch* of elements all at once, especially when using the GPU. Operations can declare they are able to process batches of elements at once using the :code:`batch` property:

.. code-block:: python

   from typing import Sequence

   @sp.register_python_op(batch=8)
   class BatchOp():
       def __init__(self, config):
           pass

       def execute(self, frame: Sequence[sp.FrameType]) -> Sequence[sp.FrameType]:
           # process a batch of frames
           ...

Notice how the signature of the :code:`execute` method changed. Since we are processing a batch of input, the :code:`frame` parameter and the output are now lists of frames instead of a single frame.

Stencil
~~~~~~~
Some operations require looking at a window of data over time. For example, computing optical flow requires both the current and next frame in time. These operations can indicate they require a :code:`stencil` of frames:

.. code-block:: python

   from typing import Sequence

   @sp.register_python_op(stencil=[0, 1])
   class OpticalFlow():
       def __init__(self, config):
           pass

       def execute(self, frame: Sequence[sp.FrameType]) -> sp.FrameType:
           # process a window of frames
           ...

Like with the :code:`batch` property, the signature of the :code:`execute` method changed. However, instead of both the input and output becoming lists of frames, only the input did. This is because the operation needs a list of frames as input, but still produces a single output element for each invocation of :code:`execute`.


Defining an Operation in C++
----------------------------
For performance critical operations, Scanner also supports defining operations in C++. Check out the tutorial `08_defining_cpp_ops.py <https://github.com/scanner-research/scanner/blob/master/examples/tutorials/08_defining_cpp_ops.py>`__ to find out how to write your own C++ operation.

..
    Bounded and Unbounded State
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    - rates
    - config rate
    - stream rate
    - element rate
    - device type
    - batch
    - stenciling
    - un/bounded state & reset
