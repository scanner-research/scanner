from scannerpy import Database, Job
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to organize your videos into Collections.            #
################################################################################

with Database() as db:

    # Instead of ingesting each video into a table individually, we can group
    # video # tables into a single entity called a collection. Here, we create
    # a collection # called "example_collection" from the video in the previous
    # example. # Collections do not incur any runtime overhead, but are simply
    # an abstraction for more easily managing your videos.
    example_video_path = util.download_video()
    input_collection, _ = db.ingest_video_collection(
        'example_collection', [example_video_path], force=True)
    print(db.summarize())

    # You can retrieve table objects off the collection.
    table = output_collection.tables(0)

    frame = db.ops.FrameInput()
    hist = db.ops.Histogram(frame=frame)
    output_op = db.ops.Output(columns=[hist])
    # You can use a collection the same way you use a table when defining a
    # computation.
    jobs = []
    for table in input_collection.tables():
        job = Job(
            op_args={
                frame: table.column('frame'),
                output_op: table.name() + '_output'
            }
        )
        jobs.append(job)
    bulk_job = BulkJob(dag=output_op, jobs=jobs)
    output_tables = db.run(bulk_job, force=True, pipeline_instances_per_node=1)

    # You can create new collections from existing tables
    hist_collection = db.new_collection('hist_collection', output_tables)
