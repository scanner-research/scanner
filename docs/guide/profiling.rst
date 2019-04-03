.. _profiling:

Performance Profiling
=====================

The goal of this guide is to enable the reader to understand the performance of Scanner graph execution and how performance can be improved by tuning execution parameters.

Overview
--------
While executing a graph, Scanner keeps a record of how long various tasks take to complete: loading data, moving data between CPU and GPU, executing operations, saving data, etc. We call this information the *profile* of a computation graph. Getting the profile for the execution of the computation graph works as follows:

.. code-block:: python

   import scannerpy as sp
   import scannertools.imgproc

   # Set up the client
   sc = sp.Client()

   # Define a computation graph
   video_stream = sp.NamedVideoStream(sc, 'example', path='example.mp4')
   input_frames = sc.io.Input([video_stream])
   resized_frames = sc.ops.Resize(frame=input_frames, width=[640], height=[480])
   output_stream = sp.NamedVideoStream(sc, 'example-output')
   output = sc.io.Output(resized_frames, [output_stream])

   # Run the computation graph
   job_id = sc.run(output, sp.PerfParams.estimate())

   # Get the profile
   profile = sc.get_profile(job_id)

The :py:class:`~scannerpy.profiler.Profile` class contains information about how long the various parts of the execution graph took and on what processors or workers (when running Scanner with multiple machines) those parts were executed. We can visualize the profile on a timeline by writing out a trace file:

.. code-block:: python

   profile.write_trace(path='resize-graph.trace')

You can view this file by going to :code:`chrome://tracing` in the Chrome browser, clicking "load" in the top left, and selecting :code:`resize-graph.trace`. 

Trace Visualization
-------------------
Scanner leverages the Chrome browser's `trace profiler tool <https://www.chromium.org/developers/how-tos/trace-event-profiling-tool>`__ to visualize the execution of a computation graph. Below is a screenshot from viewing the  trace file generated with the above code:

.. image:: /_static/trace_viz.jpg

Let's walkthrough how Scanner's execution of a graph is visualized in this interface. But first, we need to explain how Scanner actually executes a graphs. 

Scanner's Execution Model
-------------------------
Scanner operates under a master-worker model: a centralized coordinator process (the *master*) listens for commands from a client (:py:class:`scannerpy.client.Client`) and issues processing tasks to worker processes that are running on potentially hundreds of machines  (the *workers*). Naturally, the trace visualization groups together processing events by the process they occur in:

.. image:: /_static/trace_processes.jpg

Here, you can see the master process and a sole worker (there is only one worker for this trace because we ran Scanner locally only on a single machine). The events for the master process mainly deal with distributing work among the workers and tracking job progress. Since these events are usually not performance critical, we won't go into them here. On the other hand, the recorded events for a worker process reflect the execution of the computation graph that was submitted to Scanner and so are useful for understanding the performance of a Scanner graph. Let's look at the worker process' timeline of events:

.. image:: /_static/trace_worker.jpg

Within a single worker process, Scanner implements a *replicated pipeline execution model* to achieve parallelism across CPUs and GPUs. At a high level, pipeline execution breaks up the computation graph into distinct *pipeline stages*, which are each responsible for executing a subset of the graph, and these stages are hooked together using queues. For example, in the above trace screenshot the rows correspond to distinct pipeline stages: :code:`Reader[X]`, :code:`Pipeline[X]:DecodeVideo`, :code:`Pipeline[X]:Ops[Y]`, etc. Each row contains the trace events for the labeled pipeline stage. The number inside the square brackets, :code:`[X]`, represents a replica of that pipeline stage. Each replica of a pipeline stage can process data in the graph independently, and thus the number of replicas controls the amount of parallelism for a pipeline stage.

Here's a list of all the pipeline stages in a Scanner graph that will show up in the profile event trace:

- :code:`Reader[X]`: responsible for reading data from the input stored streams to the graph.
- :code:`Pipeline[X]:DecodeVideo`: responsible for decoding compressed video streams from readers.
- :code:`Pipeline[X]:Ops[Y]`: handles executing a portion of the computation graph. If a computation graph has operations that use different devices (CPU or GPU), then Scanner will group operations together based on device type and place them in a separate pipeline stage. These separate stages are indicated by the :code:`Y` index in :code:`Ops[Y]`.
- :code:`Pipeline[X]:EncodeVideo`: responsible for compressing streams of video frames destined for an output operation.
- :code:`Writer[X]`: responsible for writing data from the outputs of the graph to the output stored streams.

Let's expand the three :code:`Pipeline[0]` stages to get a handle on where the most time is being spent in this graph:

.. image:: /_static/trace_worker_expand.jpg

Notice how the timeline for :code:`Pipeline[0]:DecodeVideo` is constantly occupied by processing events, while :code:`Pipeline[0]:Ops[0]` has a large amount of idle time (indicated by the gray "Wait on Input Queue" events). This immediately tells us is that this computation graph is *decode-bound*: the throughput of the computation graph is limited by how fast we can decode frames from the input video.

Tunable Parameters
------------------
Scanner provides a collection of tunable parameters that affect the performance of executing a computation graph. Setting these parameters optimally depends on the computational properties of the graph, such as the I/O versus compute balance, usage of GPUs or CPUs, intermediate data element size, size of available memory, and latency to data storage. The tunable parameters are captured in the :code:`PerfParams` class:

.. autoclass:: scannerpy.common.PerfParams


For new users to Scanner, we recommend using the :py:meth:`scannerpy.common.PerfParams.estimate` method to have Scanner automatically set these parameters based upon inferred properties of your computation graph. More advanced users that understand how these parameters influence performance can make use of :py:meth:`scannerpy.common.PerfParams.manual` to tune the parameters themselves.

