.. _ops:

Operations
==========
Processing data in Scanner occurs primarily through *operations* (*ops*). Operations are built-in or user defined functionality that transform streams of input elements to streams of output elements.

Defining an Operation in Python
-------------------------------
Scanner supports defining operations in Python using the following syntax:

.. code-block:: python

   from scannerpy import FrameType
   import scannerpy
                
   @scannerpy.register_python_op()
   class PersonDetectorDNN(scannerpy.Kernel):
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
    
       def execute(self, frame: FrameType) -> BoundingBoxList:
           person_boxes, confidences = self.model.execute(frame)
           return [box
                   for box, conf in zip(person_boxes, confidences)
                   if conf > self._threshold]

We now have an operation defined named :code:`PersonDetectorDNN` which can be instantiated using a client object named :code:`sc` like this: :code:`sc.ops.PersonDetectorDNN(frame=..., model_url=..., threshold=[...])`. The rest of this guide will explain the above piece of code in more detail. First, let's start with the parameters to the operation.

Parameter Rates
--------------
Parameters to operations vary at one of three different *rates*. A rate is the frequency at which the parameters to the operation change. 

Init Config rate
~~~~~~~~~~~~~~~~
This is the "slowest" parameter rate to an operation. Init config rate parameters are specified once and given to the constructor of an operation, and then never changed. For example, the :code:`model_url` parameter to the :code:`PersonDetectorDNN` operation is a init config rate parameter as it is specified in the :code:`__init__` method to the class.

Stream Config Rate
~~~~~~~~~~~~~~~~~~
Stream config rate parameters are provided to an operation once before processing elements from a specific stream. These parameters parameterize the list of streams that an operation processes. For example, the :code:`threshold` parameter specified in the :code:`new_stream` method of the :code:`PersonDetectorDNN` operation is a stream config rate parameter.

Stream Rate
~~~~~~~~~~~
Stream rate parameters are parameters that bind to streams and are processed element at a time by the operation. For the :code:`PersonDetectorDNN` operation, the :code:`frame` parameter specified in the :code:`execute` method is a stream rate parameter.

Fetching Resources
------------------
Some operations require external resources to be download or fetched before they can start processing data. In the case of the :code:`PersonDetectorDNN` operation, it requires the model weights for its deep neural network. To download external resources, operations should implement the :py:meth:`scannerpy.kernel.Kernel.fetch_resources` and :py:meth:`scannerpy.kernel.Kernel.setup_with_resources` methods. To avoid redownloading resources that are shared across instances of an operation, the :py:meth:`~scannerpy.kernel.Kernel.fetch_resources` method is called once per instance of an operation on a machine to download resources. Once :py:meth:`~scannerpy.kernel.Kernel.fetch_resources` has been executed, the :py:meth:`~scannerpy.kernel.Kernel.setup_with_resources` method is called for every instance of that operation after :code:`__init__`.

Operation Properties
--------------------
Scanner operations can be annotated with several different properties to change their functionality.

Device Sets
~~~~~~~~~~~
By default, Scanner will assume operations only use the CPU when processing data. If an operation utilizes the GPU when processing elements, it can declare that it requires that device type during op declaration:

.. code-block:: python

   @scannerpy.register_python_op(device_sets=[(DeviceType.GPU, 1)])
   class GpuOp():
       def __init__(self, config):
           pass

       ...

Batch
~~~~~
Many operations benefit from being able to process a *batch* of elements all at once, especially when using the GPU. Operations can declare they are able to process batches of elements at once using the :code:`batch` property:

.. code-block:: python

   from typing import Sequence

   @scannerpy.register_python_op(batch=8)
   class BatchOp():
       def __init__(self, config):
           pass

       def execute(self, frame: Sequence[FrameType]) -> Sequence[FrameType]:
           # process a batch of frames
           ...

Notice how the signature of the :code:`execute` method changed. Since we are processing a batch of input, the :code:`frame` parameter and the output are now lists of frames instead of single frames.

Stencil
~~~~~~~
Some operations require looking at a window of data over time. For example, computing optical flow requires both the current and next frame in time. These operations can indicate they require a :code:`stencil` of frames:

.. code-block:: python

   from typing import Sequence

   @scannerpy.register_python_op(stencil=[0, 1])
   class OpticalFlow():
       def __init__(self, config):
           pass

       def execute(self, frame: Sequence[FrameType]) -> FrameType:
           # process a window of frames
           ...

Like with the :code:`batch` property, the signature of the :code:`execute` method changed. However, instead of both the input and output becoming lists of frames, only the input did. This is because the operation needs a list of frames as input, but still produces a single output element for each invocation of :code:`execute`.

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
