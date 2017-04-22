from scannerpy import Database, Job
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to organize your videos into Collections.            #
################################################################################

with Database() as db:

    # Instead of ingesting each video into a table individually, we can group video
    # tables into a single entity called a collection. Here, we create a collection
    # called "example_collection" from the video in the previous example.
    # Collections do not incur any runtime overhead, but are simply an abstraction
    # for more easily managing your videos.
    example_video_path = util.download_video()
    input_collection, _ = db.ingest_video_collection(
        'example_collection', [example_video_path], force=True)
    print(db.summarize())

    # You can use a collection the same way you use a table when defining a
    # computation. This will run your computation over every table in the
    # collection using the sampling mode you specify.
    frame = input_collection.as_op().range(0, 100)
    histogram = db.ops.Histogram(frame = frame)
    job = Job(columns = [histogram], name = 'example_hist_collection')
    output_collection = db.run(job, force=True)

    # You can retrieve table objects off the collection.
    output_table = output_collection.tables(0)
