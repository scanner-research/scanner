from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
import numpy as np

class Constants:
    def __init__(self, iw, ih, T):
        self.iw = iw
        self.ih = ih
        self.T =T
        self.d = math.floor(math.sqrt(self.ih**2 + self.iw**2))
        self.tau_c = 0.1 * self.d
        self.gamma = 0.5 * self.d

    w = 16
    g = 4
    lam_s = 200
    lam_a = 80
    tau_s = 200
    tau_a = 200

    # Speedup should be user defined
    v = 3


with Database(debug=True) as db:
    def create_database():
        db.ingest_videos([('example', '/home/wcrichto/.deps/hyperlapse/long.mp4')],
                         force=True)

    def extract_features():
        op = db.ops.FeatureExtractor(
            feature_type=db.protobufs.SURF,
            device=DeviceType.GPU)

        db.run(db.sampler().all([('example', 'example_surf')]),
               op, force=True)

    def compute_matches():
        input = db.ops.Input(["index", "features", "keypoints", "frame_info"])
        op = db.ops.FeatureMatcher(
            inputs=[(input, ["features", "keypoints", "frame_info"])],
            device=DeviceType.GPU)

        tasks = db.sampler().all([('example_surf', 'example_matches')],
                                 warmup_size=24)
        sample = tasks[0].samples.add()
        sample.table_name = "example"
        sample.column_names.extend(["frame_info"])
        sample.sampling_function = "All"
        args = db.protobufs.AllSamplerArgs()
        args.sample_size = 1000
        args.warmup_size = 24
        sample.sampling_args = args.SerializeToString()

        db.run(tasks, op, force=True)

    def build_path():
        matches = db.table('example_matches')
        T = matches.num_rows

        C = Constants(1080, 1920, T)
        Cm = np.zeros((C.T+1, C.T+1))
        rows = matches.load(['matches'], parsers.array(np.float32))
        for i, row in rows:
            Cm[i, :] = row




    # create_database()
    # extract_features()
    compute_matches()
    # build_path()
