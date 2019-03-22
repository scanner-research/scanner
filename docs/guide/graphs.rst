.. _graphs:

Computation Graphs
==================

Overview
--------
Scanner represents applications as *computation graphs*. The nodes in computation graphs are Scanner operations (check out the :ref:`operations` guide for more information) and the edges between operations represent streams of data consumed and produced by operations. Nodes in a computation graph are one of four types:

- **Input nodes** (:code:`cl.io.Input`): read data from stored streams (:ref:`stored-streams`), such as from videos or previously generated metadata. 
- **Operation nodes** (:code:`cl.ops.XXX`): represent functions that transform their inputs into new outputs, such as performing a resize of a frame. 
- **Stream operation nodes** (:code:`cl.streams.XXX`):
- **Output nodes** (:code:`cl.io.Output`): write data to empty stored streams.

For example, let's look at a computation graph with an operation that resizes frames:

.. code-block:: python

   import scannerpy as sp
   import scannertools.imgproc

   cl = sp.Client()
   video_stream = sp.NamedVideoStream(cl, 'example', path='example.mp4')
   input_frames = cl.io.Input([video_stream])
   resized_frames = cl.ops.Resize(frame=input_frames, width=[640], height=[480])
   output_stream = sp.NamedVideoStream(cl, 'example-output')
   output = cl.io.Output(resized_frames, [output_stream])

Here, the :code:`cl.io.Input`, :code:`cl.ops.Resize`, and :code:`cl.io.Output` operations are nodes in a three node graph. :code:`cl.ops.Resize` is connected to :code:`cl.io.Input` through passing its output, :code:`input_frames`, to the resize operation, :code:`frame=input_frames`. Likewise for :code:`cl.io.Output` and :code:`cl.ops.Resize`, but :py:meth:`~scannerpy.io.IOGenerator.Output` operations also bind an edge from a computation graph to an empty stored stream (:code:`output_stream` here) to be filled in with the sequence of elements produce from that edge. Importantly, note that we have not processed any data at this point: we have only defined a computation graph that we can tell Scanner to execute. Let's do that now:

.. code-block:: python

   cl.run(output)

This operation will kick-off a Scanner job that will read all the elements in the input :code:`video_stream` and write outputs to :code:`output_stream`. 

The rest of this guide goes into further detail on the capabilities of computation graphs.

Multiple inputs and outputs
---------------------------
Computation graphs can have any number of inputs and outputs. Here's an example graph with one input and two outputs:

.. code-block:: python

   video_stream = sp.NamedVideoStream(cl, 'example', path='example.mp4')
   input_frames = cl.io.Input([video_stream])

   large_frames = cl.ops.Resize(frame=input_frames, width=[1280], height=[720])
   small_frames = cl.ops.Resize(frame=input_frames, width=[640], height=[480])

   large_stream = sp.NamedVideoStream(cl, 'large-output')
   large_output = cl.io.Output(large_frames, [large_stream])

   small_stream = sp.NamedVideoStream(cl, 'small-output')
   small_output = cl.io.Output(small_frames, [small_stream])

   cl.run([large_output, small_output])

Notice how we pass both outputs to the :py:meth:`~scannerpy.client.Client.run` method. Scanner only runs the portions of the graph needed to produce the streams for the outputs passed to :code:`run`.

.. _batch-processing:

Batch processing of stored streams
----------------------------------
Often, one has a large collection of videos that they want to run the same computation graph over. Scanner supports this via batch processing of input and output streams. To process a batch of streams, create a list of :ref:`stored-streams` representing the input videos and then pass that list to the input operation:

.. code-block:: python

   input_streams = [
       NamedVideoStream(cl, 'example1', path='example1.mp4'),
       NamedVideoStream(cl, 'example2', path='example2.mp4'),
       ...
       NamedVideoStream(cl, 'example100', path='example100.mp4')]
   input_frames = cl.io.Input(input_streams)
   resized_frames = cl.ops.Resize(frame=input_frames,
                                  width=[640, 1280, ..., 480],
                                  height=[480, 720, ..., 360])

Note that this is different from having multiple inputs or outputs to a computation graph. This graph still has only  one input because each video in the batch is processed independently. Conceptually, you can think of batch processing as executing a separate instance of the graph for each input stream in a batch. Notice the other change that we made to this graph: the :code:`width` and :code:`height` arguments to :code:`Resize` are now lists of the same length as :code:`input_streams`. This is because :code:`height` and :code:`width` are *stream config parameters* to :code:`Resize`: each input stream gets its own set of parameters. Check out the :ref:`stream-config-parameters` section to learn more about how stream config parameters work.

We also need a corresponding output stream for each of our input streams:

.. code-block:: python

   output_streams = [
       NamedVideoStream(cl, 'example1-resized'),
       NamedVideoStream(cl, 'example2-resized'),
       ...
       NamedVideoStream(cl, 'example100-resized')]
   output = cl.io.Output(resized_frames, output_streams)

When executing this graph, Scanner will read and process each input stream independently to produce the output streams. If Scanner is running on a multi-core machine, multi-GPU machine, or on a cluster of machines, the videos will be processed in parallel across any of those configurations.

.. _stream-operations:

Stream Operations
-----------------
Most operations are restricted to produce a single output element for each input element they receive. However, sometimes an application only needs to process a subset of all of the input elements from a stored stream. Scanner supports this using *stream operations*. For example, if an application only requires every third frame from a video, we can use a :py:meth:`~scannerpy.streams.StreamsGeneator.Stride` operation:

.. code-block:: python

   input_frame = cl.io.Input([video_stream])
   resized_frame = cl.ops.Resize(frame=input_frame, width=[640], height=[480])
   sampled_frame = cl.streams.Stride(resized_frame, [3])

If :code:`video_stream` is of length 30, then :code:`sampled_frame` will be a sequence of length 10 with the frames at indices [0, 3, 6, 9, ... 27]. Scanner also supports other types of stream operations, such as :py:meth:`~scannerpy.streams.StreamsGeneator.Gather`, which selects frames given a list of indices:

.. code-block:: python

   sampled_frame = cl.streams.Gather(resized_frame, [[0, 5, 7, 29]])

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
