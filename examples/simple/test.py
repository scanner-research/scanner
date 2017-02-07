from scannerpy import Database

db = Database()

input_videos = db.ingest_video_collection(
    'meangirls', ['/bigdata/wcrichto.new/videos/meanGirls.mp4'])

hist_evaluator = db.evaluators.Histogram()

output_histograms = db.run(videos, hist_evaluator, 'meangirls_histogram')

histogram = output_histograms.tables[0].columns[0].load()
