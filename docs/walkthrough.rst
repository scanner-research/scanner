.. _walkthrough:

Walking through a complete app
==============================

To explain how Scanner is used, let's walk through a simple example that reads every third frame from a video, resizes the frames, and then creates a new video from the sequence of resized frames.

.. note::

   This Quickstart walks you through a very basic Scanner application that downsamples a video in space and time. Once you are done with this guide, check out the `examples <https://github.com/scanner-research/scanner/blob/master/examples>`__ directory for more useful applications, such as using Tensorflow `for detecting objects in all frames of a video <https://github.com/scanner-research/scanner/blob/master/examples/apps/object_detection_tensorflow>`__ and Caffe for `face detection <https://github.com/scanner-research/scanner/blob/master/examples/apps/face_detection>`__.

To run the code discussed here, install Scanner (:ref:`installation`). Then from the top-level Scanner directory, run:

.. code-block:: bash

   cd examples/apps/quickstart
   wget https://storage.googleapis.com/scanner-data/public/sample-clip.mp4
   python3 main.py

After :code:`main.py` exits, you should now have a resized version of :code:`sample-clip.mp4` named :code:`sample-clip-resized.mp4` in the current directory. Let's see how that happened by looking inside :code:`main.py`.

Starting up Scanner
-------------------
The first step in any Scanner program is to create a :py:class:`~scannerpy.database.Database` object. The :py:class:`~scannerpy.database.Database` object manages videos or other data that you have may have stored from data processing you've done in the past. The Database object also provides the API to construct and execute new video processing jobs.

.. code-block:: python

   from scannerpy import Database, Job

   db = Database()

Ingesting a video into the Database
-----------------------------------
Scanner is designed to provide fast access to frames in videos, even under random access patterns. In order to provide this functionality, Scanner first needs to analyze the video to build an index on the video. For example, given an mp4 video named :code:`example.mp4`, we can ingest this video as follow:

.. code-block:: python

   db.ingest_videos([('table_name', 'example.mp4')])

Scanner analyzes the file to build the index and creates a :py:class:`~scannerpy.table.Table` for that video in the Scanner database called :code:`table_name`. You can see the contents of the database by running:

.. code-block:: python

   >>> print(db.summarize())
                         ** TABLES **
   ---------------------------------------------------
   ID | Name       | # rows | Columns      | Committed
   ---------------------------------------------------
   0  | table_name | 360    | index, frame | true

By default, ingest copies the video data into the Scanner database (located at :code:`~/.scanner/db` by default). However, Scanner can also read videos without copying them using the :code:`inplace` flag.

.. code-block:: python

   db.ingest_videos([('table_name', 'example.mp4')], inplace=True)

This still builds the index for accessing the video but avoids copying the files
into the database.

.. _defining_a_graph:

Defining a Computation Graph
----------------------------

Now we can tell Scanner how to process the video by constructing a *computation graph*. A computation graph is a graph of input nodes (**Sources**), function nodes (**Ops**), and output nodes (**Sinks**). **Sources** can read data from the Scanner database (such as the table we ingested above) or from other sources of data, such as the filesystem or a SQL database. **Ops** represent functions that transform their inputs into new outputs. **Sinks**, like **Sources**, write data to the database or to other forms of persistent storage.

Let's define a computation graph to read frames from the database, select every third frame, resize them to 640 x 480 resolution, and then save them back to a new database table. First, we'll create a Source that reads from a column in a table:

.. code-block:: python

   frame = db.sources.FrameColumn()

But wait a second, we didn't tell the **Source** the table and column it should read from. What's going on? Since it's fairly typical to use the same computation graph to process a collection of videos at once, Scanner adopts a "binding model" that lets the user define a computation graph up front and then later "bind" different videos to the inputs. We'll see this in action in the :ref:`defining_a_job` section.

The :code:`frame` object returned by the **Source** represents the stream of frames that are stored in the table, and we'll use it as the input to the next operation:

.. code-block:: python

   sampled_frame = db.streams.Stride(input=frame, stride=3) # Select every third frame

This is where we select only every third frame from the stream of frames we read from the **Source**. This comes from a special class of ops (from :code:`db.streams`) that can change the size of a stream, as opposed to transforming inputs to outputs 1-to-1.

We then process the sampled frames by instantiating a Resize **Op** that will resize the frames in the :code:`frame` stream to 640 x 480:

.. code-block:: python

   resized = db.ops.Resize(frame=sampled_frame, width=640, height=480)

This **Op** returns a new stream of frames which we call :code:`resized`. The Resize **Op** is one of the collection of built-in **Ops** in the :ref:`standard_library`. (You can learn how to write your own **Ops** by following the :ref:`tutorial`.)

Finally, we write these resized frames to a column called 'frame' in a new table by passing them into a column **Sink**:

.. code-block:: python

   output_frame = db.sinks.Column(columns={'frame': resized})

Putting it all together, we have:

.. code-block:: python

   frame = db.sources.FrameColumn()
   sampled_frame = db.streams.Stride(input=frame, stride=3)
   resized = db.ops.Resize(frame=sampled_frame, width=640, height=480)
   output_frame = db.sinks.Column(columns={'frame': resized})

At this point, we have defined a computation graph that describes the computation to run, but we haven't yet told Scanner to execute the graph.

.. _defining_a_job:

Defining Jobs
-------------

As alluded to in :ref:`defining_a_graph`, we need to tell Scanner which table we should read and which table we should write to before executing the computation graph. We can perform this "binding" of arguments to graph nodes using a **Job**:

.. code-block:: python

   job = Job(op_args={
       frame: db.table('table_name').column('frame'),
       output_frame: 'resized_table'
   })

Here, we say that the :code:`FrameColumn` indicated by :code:`frame` should read from the column :code:`frame` in the table :code:`"table_name"`, and that the output table indicated by :code:`output_frame` should be called :code:`"resized_table"`.

Running a Job
--------------

Now we can run the computation graph over the video we ingested. This is done by simply calling :code:`run` on the database object, specifying the jobs and outputs that we are interested in:

.. code-block:: python

   output_tables = db.run(output=output_frame, jobs=[job]) 

This call will block until Scanner has finished processing the job. You should see a progress bar while Scanner is executing the computation graph. Once the job are done, :code:`run` returns the newly computed tables, here shown as :code:`output_tables`.

Reading the results of a Job
----------------------------

We can directly read the results of job we just ran in the Python code by querying the :code:`frame` column on the table :code:`resized_table`:

.. code-block:: python

   for resized_frame in db.table('resized_table').column('frame').load():
       print(resized_frame.shape)

Video frames are returned as numpy arrays. Here we are printing out the shape of the frame, which should have a width of 640 and height of 480.

Exporting to mp4
----------------

We can also directly save the frame column as an mp4 file by calling :code:`save_mp4` on the :code:`frame` column:

.. code-block:: python

   db.table('resized_table').column('frame').save_mp4('resized-video')

After this call returns, an mp4 video should be saved to the current working directory called :code:`resized-video.mp4` that consists of the resized frames that we generated.

That's a complete Scanner pipeline! If you'd like to learn about process multiple jobs, keep reading! Otherwise, to learn more about the features of Scanner, either follow the :ref:`walkthrough` or go through the extended :ref:`tutorial`.

.. toctree::
   :maxdepth: 1

Processing multiple videos
--------------------------

Now let's say that we have a directory of videos we want to process, instead of just a single one as above. 
To see the multiple video code in action, run the following commands from the quickstart app directoroy:

.. code-block:: bash

   wget https://storage.googleapis.com/scanner-data/public/sample-clip-1.mp4
   wget https://storage.googleapis.com/scanner-data/public/sample-clip-2.mp4
   wget https://storage.googleapis.com/scanner-data/public/sample-clip-3.mp4
   python3 main-multi-video.py

After :code:`main-multi-video.py` exits, you should now have a resized version of each of the downloaded videos named :code:`sample-clip-%d-resized.mp4` in the current directory, where :code:`%d` is replaced with the number of the video.

There are two places in the code that need to change to process multiple videos. Let's look at those pieces of code inside :code:`main-multi-video.py` now.

Ingesting multiple videos
-------------------------

The first change is that we need to ingest all of our videos. This means changing our call to :code:`ingest_videos` to take a list of three tuples, instead of just one:

.. code-block:: python

   videos_to_process = [
       ('sample-clip-1', 'sample-clip-1.mp4'),
       ('sample-clip-2', 'sample-clip-2.mp4'),
       ('sample-clip-3', 'sample-clip-3.mp4')
   ]
   
   # Ingest the videos into the database
   db.ingest_videos(videos_to_process)

Now we have three tables that are ready to be processed!

Defining and executing multiple Jobs
------------------------------------

The second change is to define multiple jobs, one for each video that we want to process.

.. code-block:: python

   jobs = []
   for table_name, _ in videos_to_process:
       job = Job(op_args={
           frame: db.table(table_name).column('frame'),
           output_frame: 'resized-{:s}'.format(table_name)
       })
       jobs.append(job)

Now we can process these multiple jobs at the same time using :code:`run`:

.. code-block:: python

   output_tables = db.run(output=output_frame, jobs=jobs)

Like before, this call will block until Scanner has finished processing all the jobs. You should see a progress bar while Scanner is executing the computation graph as before. Once the jobs are done, :code:`run` returns the newly computed tables, here shown as :code:`output_tables`.

Walking through a more advanced Jupyter-based app
=================================================

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
