Quickstart
==========

.. toctree::
   :maxdepth: 1

Scanner applications are written using a simple Python API:

.. code-block:: python

   from scannerpy import Database, Job

   db = Database()
   db.ingest_videos([('table_name', 'example.mp4')])

   frame = db.sources.FrameColumn()
   resized = db.ops.Resize(frame=frame, width=640, height=480)
   output_frame = db.sinks.Column(columns={'frame': resized})

   job = Job(op_args={
       frame: db.table('table_name').column('frame'),
       output_frame: 'resized_example'
   })

   output_tables = db.run(output=output, jobs=[job], force=True)
