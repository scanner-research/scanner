.. _walkthrough:

Walkthroughs
============

Walking through a simple application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To explain how Scanner is used, let's walk through a simple example that reads every third frame from a video, resizes the frames, and then creates a new video from the sequence of resized frames.

To run the code discussed here, install Scanner (:ref:`installation`). Then from the top-level Scanner directory, run:

.. code-block:: bash

   cd examples/apps/quickstart
   wget https://storage.googleapis.com/scanner-data/public/sample-clip.mp4
   python3 main.py

After :code:`main.py` exits, you should now have a resized version of :code:`sample-clip.mp4` named :code:`sample-clip-resized.mp4` in the current directory. Let's see how that happened by looking inside :code:`main.py`.

Starting up Scanner
-------------------
The first step in any Scanner program is to create a :py:class:`~scannerpy.client.Client` object. The :py:class:`~scannerpy.client.Client` object manages your connection to the Scanner runtime and provides the API to execute new video processing jobs. By default, creating a client will first start an instance of Scanner on the local machine and then establish a connection to it:

.. code-block:: python

   from scannerpy import Client, PerfParams

   sc = Client()

Reading data with stored streams
--------------------------------
Inputs and outputs to Scanner applications are represented as streams of data, called :py:class:`~scannerpy.storage.StoredStream` s (check out the :ref:`stored-streams` guide for more information). Scanner has built-in support for videos, audio, lists of binary data, sequences of files, and SQL databases. For example, let's create a stream representing a video:

.. code-block:: python

   input_stream = NamedVideoStream(sc, 'sample-clip', path='sample-clip.mp4')

:py:class:`~scannerpy.storage.NamedVideoStream` is a stream that stores data in Scanner's internal database format. In this case, we stored the data under the name :code:`sample-clip` for the video file :code:`sample-clip.mp4`.

Since Scanner was built specifically for processing video, it has specialized support for fast access to frames in videos, even under random access patterns. In order to provide this functionality, Scanner first needs to analyze the video to build an index on the video. This index is built when a :py:class:`~scannerpy.storage.NamedVideoStream` is first accessed. By default, Scanner copies the video data to Scanner's internal database (located at :code:`~/.scanner/db` by default). However, Scanner can also read videos without copying them using the :code:`inplace` flag:

.. code-block:: python

   input_stream = NamedVideoStream(sc, 'sample-clip', path='sample-clip.mp4',
                                   inplace=True)

This still builds the index for accessing the video but avoids copying the files.

.. _defining_a_graph:

Defining a Computation Graph
----------------------------

Now we can tell Scanner how to process the video by constructing a *computation graph*. A computation graph is a graph of input nodes (**Inputs**), function nodes (**Ops**), and output nodes (**Outputs**). **Inputs** read data from stored streams, such as the video stream we just created. **Ops** represent functions that transform their inputs into new outputs. **Outputs** write data to stored streams.

Let's define a computation graph to read from a video, select every third frame, resize those frames to 640 x 480 resolution, and then save them back to a new stored stream. First, we'll create an **Input** that reads from our video:

.. code-block:: python

   frame = sc.io.Input([input_stream])

The :code:`frame` object returned by **Input** represents the stream of frames that are stored in the table, and we'll use it as the input to the next operation:

.. code-block:: python

   sampled_frame = sc.streams.Stride(frame, 3) # Select every third frame

This is where we select only every third frame from the stream of frames we read from the **Source**. This comes from a special class of ops (from :code:`sc.streams`) that can change the size of a stream, as opposed to transforming inputs to outputs 1-to-1.

We then process the sampled frames by instantiating a Resize **Op** that will resize the frames in the :code:`frame` stream to 640 x 480:

.. code-block:: python

   resized = sc.ops.Resize(frame=sampled_frame, width=640, height=480)

This **Op** returns a new stream of frames which we call :code:`resized`. The Resize **Op** is one of the collection of built-in **Ops** in the :ref:`standard_library`. (You can learn how to write your own **Ops** by following the :ref:`tutorial`.)

Finally, we write these resized frames to a new stream called 'sampled-clip-resized' by passing them into an **Output**:

.. code-block:: python

   output_stream = NamedVideoStream(sc, 'sample-clip-resized')
   output = sc.io.Output(resized, [output_stream])

Putting it all together, we have:

.. code-block:: python

   input_stream = NamedVideoStream(sc, 'sample-clip', path='sample-clip.mp4')
   frame = sc.io.Input([input_stream])
   sampled_frame = sc.streams.Stride(frame, 3) # Select every third frame
   resized = sc.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
   output_stream = NamedVideoStream(sc, 'sample-clip-resized')
   output = sc.io.Output(resized, [output_stream])

At this point, we have defined a graph that describes the computation to run, but we haven't yet told Scanner to execute the graph.

.. _defining_a_job:

Executing a computation graph
-----------------------------

Executing a graph is done by calling :code:`run` on the client object, specifying the outputs we want to produce:

.. code-block:: python

   sc.run(output, PerfParams.estimate())

This call will block until Scanner has finished processing the job. You should see a progress bar while Scanner is executing the computation graph. The :py:class:`~scannerpy.common.PerfParams` are parameters used to tune the performance of graph execution, e.g. the number of video frames that should be in memory at any one time. By default, the :py:meth:`~scannerpy.common.PerfParams.estimate` guesses an appropriate value of all parameters for your graph.

Reading from a stored stream
----------------------------

We can directly read the results of executing a graph by reading from the stored streams we computed:

.. code-block:: python

   for resized_frame in output_stream.load():
       print(resized_frame.shape)

Video frames are returned as numpy arrays. Here we are printing out the shape of the frame, which should have a width of 640 and height of 480.

Exporting to mp4
----------------

We can also directly save :py:class:`~scannerpy.storage.NamedVideoStream` s as mp4 files by calling :code:`save_mp4` on the output stream:

.. code-block:: python

   output_stream.save_mp4('resized-video')

After this call returns, an mp4 video should be saved to the current working directory called :code:`resized-video.mp4` that consists of the resized frames that we generated.

That's a complete Scanner application! If you'd like to learn about process multiple jobs, keep reading! Otherwise, to learn more about the features of Scanner, checkout the following:

- :ref:`walkthrough`:
- :ref:`tutorial`: introduces each of Scanner's features with code examples.
- :ref:`stored-streams`: describes the stored stream interface.
- :ref:`graphs`: describes how computation graphs are constructed and configured.
- :ref:`ops`: describes the capabilities of Scanner's ops and how they work inside computation  graphs.

.. toctree::
   :maxdepth: 1

Processing multiple videos
--------------------------

Now let's say that we have a directory of videos we want to process, instead of just a single one as above. To see the multiple video code in action, run the following commands from the quickstart app directoroy:

.. code-block:: bash

   wget https://storage.googleapis.com/scanner-data/public/sample-clip-1.mp4
   wget https://storage.googleapis.com/scanner-data/public/sample-clip-2.mp4
   wget https://storage.googleapis.com/scanner-data/public/sample-clip-3.mp4
   python3 main-multi-video.py

After :code:`main-multi-video.py` exits, you should now have a resized version of each of the downloaded videos named :code:`sample-clip-%d-resized.mp4` in the current directory, where :code:`%d` is replaced with the number of the video.

There are two places in the code that need to change to process multiple videos. Let's look at those pieces of code inside :code:`main-multi-video.py` now.

Processing multiple stored streams
----------------------------------

Instead of passing a single stream to the **Input** op, we are going to create a stream for each of our videos and pass them all at once into the **Input**:

.. code-block:: python

   videos_to_process = [
       ('sample-clip-1', 'sample-clip-1.mp4'),
       ('sample-clip-2', 'sample-clip-2.mp4'),
       ('sample-clip-3', 'sample-clip-3.mp4')
      ]
   input_streams = [NamedVideoStream(sc, info[0], path=info[1])
                    for info in videos_to_process]
   frame = sc.io.Input(input_streams)

We also need a corresponding output stream for each input stream:

.. code-block:: python

   output_streams = [NamedVideoStream(sc, info[0] + 'resized')
                    for info in videos_to_process]
   output = sc.io.Output(resized, output_streams)

When executing this graph, Scanner will read and process each input stream independently to produce the output streams. If Scanner is running on a multi-core machine, multi-GPU machine, or on a cluster of machines, the videos will be processed in parallel across any of those configurations.

Walking through a more advanced Jupyter-based app
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get a more detailed understanding of how Scanner can be used in a real
application, we recommend trying the Jupyter notebook tutorial. To start the
notebook, if you're using Docker:

.. code-block:: bash

   pip3 install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose up cpu

If you installed Scanner yourself, then run:

.. code-block:: bash

   pip3 install jupyter requests matplotlib
   cd path/to/scanner
   jupyter notebook --ip=0.0.0.0 --port=8888

Then visit port 8888 on your server/localhost, click through to
:code:`examples/Walkthrough.ipynb`, and follow the directions in the notebook.
