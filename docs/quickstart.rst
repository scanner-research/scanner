.. _quickstart:

Quickstart
==========

To explain how you Scanner is used, let's walkthrough a simple example that
reads frames from a video, resizes them, and then creates a new video using
those resized frames.


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

Scanner can also operate on videos in-place by providing the `inplace` flag.

.. code-block:: python

   db.ingest_videos([('table_name', 'example.mp4')], inplace=True)

This avoids copying the files into the database, but still creates a table.


Defining a Computation Graph
----------------------------

Scanner executes video processing pipelines

.. code-block:: python

   frame = db.sources.FrameColumn()
   resized = db.ops.Resize(frame=frame, width=640, height=480)
   output_frame = db.sinks.Column(columns={'frame': resized})


Defining a Job
--------------

.. code-block:: python

   job = Job(op_args={
       frame: db.table('table_name').column('frame'),
       output_frame: 'resized_example'
   })


Running a Job
--------------

.. code-block:: python

   output_tables = db.run(output=output, jobs=[job], force=True)


Reading the results of a Job
----------------------------

.. code-block:: python

   for resized_frame in db.table('resized_example').column('frame'):
       print(resized_frame)


Saving a video file
-------------------

.. code-block:: python

   db.table('resized_example').column('frame').save_mp4('resized_video.mp4')


.. toctree::
   :maxdepth: 1
