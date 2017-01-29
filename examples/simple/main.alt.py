from scannerpy import Database, Evaluator, DeviceType

db = Database()

blur = db.evaluators.Blur(
    device = DeviceType.CPU,
    kernel_size = 3,
    sigma = 0.5)

def without_collection():
    videos = [('meangirls', '/bigdata/wcrichto/videos/meanGirls_short.mp4')]
    for video in videos: db.ingest_video(*video)
    table_names = list(zip(*videos)[0])
    sampler = db.make_sampler()
    tasks = sampler.all_frames(table_names)
    db.run(tasks, blur)

def with_collection():
    collection = db.ingest_video_collection(
        'meangirls', ['/bigdata/wcrichto/videos/meanGirls_short.mp4'])
    db.run(collection, blur, 'meangirls_blurred')

with_collection()
