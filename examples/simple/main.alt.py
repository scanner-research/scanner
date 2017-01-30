from scannerpy import Database, Evaluator, DeviceType

db = Database()

blur = db.evaluators.Blur(
    device = DeviceType.CPU,
    kernel_size = 3,
    sigma = 0.5)

def single_video():
    video = '/bigdata/wcrichto/videos/meanGirls_short.mp4'
    db.ingest_video(('meangirls', video))
    sampler = db.sampler()
    tasks = sampler.all_frames([('meangirls', 'meangirls_blurred')])
    db.run(tasks, blur)

def video_collection():
    input_collection = db.ingest_video_collection(
        'meangirls', ['/bigdata/wcrichto/videos/meanGirls_short.mp4'])
    # collection = db.get_collection('meangirls')
    output_collection = db.run(input_collection, blur, 'meangirls_blurred')
    table = output_collection.tables(0)
    print next(table.load_frames()).shape

video_collection()
