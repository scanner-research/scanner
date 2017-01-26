from scannerpy import Database, Evaluator, DeviceType

db = Database()
master = db.start_master()
worker = db.start_worker()

db.ingest_videos('meangirls', ['/bigdata/wcrichto/videos/meanGirls_short.mp4'])

input = Evaluator.input()
blur = Evaluator(
    'Blur',
    inputs=[(input, ['frame', 'frame_info'])],
    device=DeviceType.CPU,
    args={
        'kernel_size': 3,
        'sigma': 0.5
    })
output = Evaluator.output([(blur, 'frame')])

sampler = db.make_sampler()
tasks = sampler.all_frames('meangirls')

db.run(tasks, output, 'meangirls_blurred')
