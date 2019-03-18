.. _graphs:

Computation Graphs
==================
Scanner represents applications as *computation graphs*. The nodes in computation graphs are Scanner operations (check out the :ref:`operations` guide for more information) and the edges between operations represent streams of data consumed and produced by operations. For example, let's look at a partial computation graph that with an operation that resizes frames:

.. code-block:: python

   from scannerpy import Client
   from scannerpy.storage import NamedVideoStream
   sc = Client()
   video_stream = NamedVideoStream(sc, 'example', path='example.mp4')
   input_frame = sc.io.Input([video_stream])
   resized_frame = sc.ops.Resize(frame=input_frame, width=[640], height=[480])

Here, the :code:`sc.io.Input` and :code:`sc.ops.Resize` operations are both nodes in a two node graph. :code:`sc.ops.Resize` is connected to :code:`sc.io.Input` by passing the output of the input operation, :code:`input_frame`, as an input to the resize operation, :code:`frame=input_frame`. It's important to note that we have not processed any data at this point. We have only defined a partial computation graph that we can finish later and then tell Scanner to execute.

Processing lists of stored streams
----------------------------------
The partial computation graph defined above only processes one video, but Scanner supports defining graphs that operate on hundreds or even thousands of videos at once. To process multiple videos, create a list of :ref:`stored-streams` representing those videos and then pass that list to the input operation:

.. code-block:: python

   video_streams = [
       NamedVideoStream(sc, 'example1', path='example1.mp4'),
       NamedVideoStream(sc, 'example2', path='example2.mp4'),
       ...
       NamedVideoStream(sc, 'example100', path='example100.mp4')]
   input_frame = sc.io.Input(video_streams)
   resized_frame = sc.ops.Resize(frame=input_frame,
                                 width=[640, 640, ..., 640],
                                 height=[480, 480, ..., 480])

Notice the other change that we made to this graph: the :code:`width` and :code:`height` arguments to :code:`Resize` are now lists of the same length as :code:`video_streams`. This is because :code:`height` and :code:`width` are *stream rate* arguments to  :code:`Resize`. Check out the section on *Parameter Rates* in the :ref:`ops` guide to learn more.

Executing a computation graph
-----------------------------
Once a complete graph has been defined, we can execute it using a Scanner client object. A graph is considered complete when it has at least one :py:meth:`~scannerpy.io.IOGenerator.Output` operation:

.. code-block:: python

   input_stream = NamedVideoStream(sc, 'example', path='example.mp4')
   input_frame = sc.io.Input([input_stream])
   resized_frame = sc.ops.Resize(frame=input_frame, width=[640], height=[480])
   output_stream = NamedVideoStream(sc, 'example-output')
   output = sc.io.Output(resized_frame, [output_stream])

:py:meth:`~scannerpy.io.IOGenerator.Output` operations bind an edge from a computation graph to an empty stored stream to be filled in with the sequence of elements produce from that edge. Now that we have a complete graph, we can execute it:

.. code-block:: python

   sc.run(output)

This operation will kick-off a Scanner job that will read all the elements in the input stored streams and write outputs to the stored streams provided to the output operation. Scanner also supports saving multiple outputs:

.. code-block:: python

   resized_stream = NamedVideoStream(sc, 'resized-example-output')
   resized_output = sc.io.Output(resized_frame, [resized_stream])
   frame_stream = NamedVideoStream(sc, 'frame-example-output')
   frame_output = sc.io.Output(input_frame, [frame_stream])
   sc.run([resized_output, frame_output])

Stream Operations
-----------------
Most operations are restricted to produce a single output element for each input element they receive. However, sometimes an application only needs to process a subset of all of the input elements from a stored stream. Scanner supports this using *stream operations*. For example, if an application only requires every third frame from a video, we can use a :py:meth:`~scannerpy.streams.StreamsGeneator.Stride` operation:

.. code-block:: python

   input_frame = sc.io.Input([video_stream])
   resized_frame = sc.ops.Resize(frame=input_frame, width=[640], height=[480])
   sampled_frame = sc.streams.Stride(resized_frame, [3])

If :code:`video_stream` is of length 30, then :code:`sampled_frame` will be a sequence of length 10 with the frames at indices [0, 3, 6, 9, ... 27]. Scanner also supports other types of stream operations, such as :py:meth:`~scannerpy.streams.StreamsGeneator.Gather`, which selects frames given a list of indices:

.. code-block:: python

   sampled_frame = sc.streams.Gather(resized_frame, [[0, 5, 7, 29]])

To see the full list of stream operations, check out the methods of :py:class:`~scannerpy.streams.StreamsGeneator`.

..     
    Slicing Operations
    ------------------
    In addition to stream operations, Scanner also supports special *slicing operations*.
    
    .. code-block:: python
    
       input_frame = sc.io.Input(video_streams)
       sampled_frame = sc.streams.Slice(resized_frame, 3)
       resized_frame = sc.ops.Resize(frame=input_frame,
                                     width=[640, 640, ..., 640],
                                     height=[480, 480, ..., 480])
       sampled_frame = sc.streams.Unslice(resized_frame, 3)

    - Nodes and edges
    - Stream operations
    - Multiple inputs/output streams
    - Slicing
    - Argument binding
