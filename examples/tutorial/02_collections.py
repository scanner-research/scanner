from scannerpy import Database
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to organize your videos into Collections.            #
################################################################################

with Database() as db:
    hist_op = db.ops.Histogram()

    # Instead of ingesting each video into a table individually, we can group video
    # tables into a single entity called a collection. Here, we create a collection
    # called "example_collection" from the video in the previous example.
    # Collections do not incur any runtime overhead, but are simply an abstraction
    # for more easily managing your videos.
    example_video_path = util.download_video()
    input_collection, _ = db.ingest_video_collection(
        'example_collection', [example_video_path], force=True)
    print(db.summarize())

    # We can also provide collections directly to the run function which will run
    # the op over all frames in all videos in the collection.
    output_collection = db.run(input_collection, hist_op, 'example_hist_collection',
                               force=True)

    # You can retrieve table objects off the collection.
    output_table = output_collection.tables(0)
