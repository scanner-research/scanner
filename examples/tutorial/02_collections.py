from scannerpy import Database, CollectionJob, TableJob
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

    jobs = []
    for frame, frame_info in input_collection.as_op().range(0, 100):
        histogram = db.ops.Histogram(frame = frame, frame_info = frame_info)
        jobs.append(TableJob(columns=[histogram]))
    job = CollectionJob(jobs = jobs, name = 'example_hist_collection')

    # We can also provide collections directly to the run function which will run
    # the op over all frames in all videos in the collection.
    output_collection = db.run(job, force=True)

    # You can retrieve table objects off the collection.
    output_table = output_collection.tables(0)
