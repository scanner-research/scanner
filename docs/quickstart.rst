.. _quickstart:

Quickstart
==========

To explain how Scanner is used, let's walkthrough a simple example that
reads frames from a video, selects every third frame, resizes them, and then
creates a new video using those resized frames. If you'd like to run the code
first, install Scanner (:ref:`installation`) and from the top-level Scanner
directory, run:

.. code-block:: bash

   cd examples/apps/quickstart
   wget https://storage.googleapis.com/scanner-data/public/sample-clip.mp4
   python3 main.py

After "main.py" exits, you should now have a resized version of
"sample-clip.mp4" named "resized-video.mp4" in the current directory. Let's see
how that happened by looking inside "main.py".

Starting up Scanner
-------------------
The first step in any Scanner program is to create a `Database` object.
The `Database` object manages videos or other data that you have may have
stored from data processing you've done in the past. The Database object also
provides the API to construct and execute new video processing jobs.

.. code-block:: python

   from scannerpy import Database, Job

   db = Database()

Ingesting a video into the Database
-----------------------------------
Scanner is designed to provide fast access to frames in videos, even under
random access patterns. In order to provide this functionality, Scanner first
needs to analyze the video to build an index on the video. For example, given
an mp4 video named 'example.mp4', we can ingest this video as follow:

.. code-block:: python

   db.ingest_videos([('example_table', 'example.mp4')])

Scanner analyzes the file to build the index and creates a *table* for that
video in the database. You can see the contents of the database by running:

.. code-block:: python

   >>> print(db.summarize())
                         ** TABLES **
   ---------------------------------------------------
   ID | Name       | # rows | Columns      | Committed
   ---------------------------------------------------
   0  | table_name | 360    | index, frame | true

Scanner can also read videos without copying them using the `inplace` flag.

.. code-block:: python

   db.ingest_videos([('table_name', 'example.mp4')], inplace=True)

This still builds the index for accessing the video but avoids copying the files
into the database.

.. _defining_a_graph:

Defining a Computation Graph
----------------------------

Now we can tell Scanner how to process the video by constructing a
*computation graph*. A computation graph is a graph of input nodes (**Sources**),
function nodes (**Ops**), and output nodes (**Sinks**). **Sources** can read data from
the database (such as the table we ingested above) or from other sources of
data, such as the filesystem. **Ops** represent functions that transform their
inputs into new outputs. **Sinks**, like **Sources**, write data to the database or to
other forms of persistent storage.

Let's define a computation graph to read frames from the database, select every
third frame, resize them to 640 x 480 resolution, and then save them back to a new
database table. First, we'll create a Source that reads from a column in a table:

.. code-block:: python

   frame = db.sources.FrameColumn()

But wait a second, we didn't tell the **Source** the table and column it should
read from. What's going on? Since it's fairly typical to use the same
computation graph to process a collection of videos at once, Scanner adopts a
"binding model" that lets the user define a computation graph up front and then
later "bind" different videos to the inputs. We'll see this in action in the
:ref:`defining_a_job` section.

The :code:`frame` object returned by the **Source** represents the stream of frames that
are stored in the table, and we'll use it as the input to the next operation:

.. code-block:: python

   sampled_frame = db.ops.Stride(frame, 3) # Select every third frame

This is where we select only every third frame from the stream of frames we read
from the **Source**.

We then process the sampled frames by instantiating a Resize **Op** that will
resize the frames in the :code:`frame` stream to 640 x 480:

.. code-block:: python

   resized = db.ops.Resize(frame=sampled_frame, width=640, height=480)

This **Op** returns a new stream of frames which we call :code:`resized`. The Resize
**Op** is one of the collection of built-in **Ops** in the
:ref:`standard_library`. (You can learn how to write your own **Ops** by following the :ref:`tutorial`.)

Finally, we write these resized frames to a column called 'frame' in a new table
by passing them into a column **Sink**:

.. code-block:: python

   output_frame = db.sinks.Column(columns={'frame': resized})

Putting it all together, we have:

.. code-block:: python

   frame = db.sources.FrameColumn()
   sampled_frame = db.ops.Stride(frame, 3) # Select every third frame
   resized = db.ops.Resize(frame=sampled_frame, width=640, height=480)
   output_frame = db.sinks.Column(columns={'frame': resized})

At this point, we have defined a computation graph that describes the
computation to run, but we haven't yet told Scanner to execute the graph.

.. _defining_a_job:

Defining a Job
--------------

As alluded to in :ref:`defining_a_graph`, we need to tell Scanner which table
we should read, how to sample the input frames, and which table we should write
to to execute the computation graph. We can perform this "binding" of arguments
to graph nodes using a **Job**:

.. code-block:: python

   job = Job(op_args={
       frame: db.table('table_name').column('frame'),
       output_frame: 'resized_table'
   })

Here, we say that the :code:`FrameColumn` indicated by :code:`frame` should read from the
column 'frame' in the table 'table_name', and that the output table indicated by
:code:`output_frame` should be called 'resized_table'.

In this example, we are only defining one job since we only have one video, but
Scanner can process multiple jobs at the same time (given they use the same
computation graph). When many jobs are provided, Scanner will process them all
in parallel.

Running a Job
--------------

Now we can run the computation graph over the video we ingested. This is done by
simply calling :code:`run` on the database object, specifying the jobs and outputs
that we are interested in:

.. code-block:: python

   output_tables = db.run(output=output_frame, jobs=[job])

This call will block until Scanner has finished processing the job. You should
see a progress bar while Scanner is executing the computation graph. Once the
jobs are done, :code:`run` returns the newly computed tables, here shown as
:code:`output_tables`.

Reading the results of a Job
----------------------------

We can directly read the results of job we just ran in our Python code by
querying the 'frame' column on the table 'resized_table':

.. code-block:: python

   for resized_frame in db.table('resized_table').column('frame').load():
       print(resized_frame.shape)

Video frames are returned as numpy arrays. Here we are printing out the shape
of the frame, which should have a width of 640 and height of 480.

Saving a video file
-------------------

We can also directly save the frame column as an mp4 file by calling
`save_mp4` on the 'frame' column:

.. code-block:: python

   db.table('resized_table').column('frame').save_mp4('resized-video')

After this call returns, an mp4 video should be saved to the current working
directory called 'resized-video.mp4' that consists of the resized frames
that we generated.

.. toctree::
   :maxdepth: 1
