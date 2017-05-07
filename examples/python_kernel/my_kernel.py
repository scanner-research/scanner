import numpy as np
import tensorflow as tf

class Kernel:
    def __init__(self):
        print 'Init'
        # self._sess = tf.Session()

    def close(self):
        pass
        # self._sess.close()

    def execute(self, cols):
        print 'Execute'
        frame = cols[0]
        tensor = tf.stack(frame)
        planes = tf.split(tensor, 3, axis=2)
        value_range = tf.constant([0.0, 255.0], dtype=tf.float64)
        hists = [tf.histogram_fixed_width(
            tf.cast(plane, tf.float64), value_range, nbins=16)
            for plane in planes]
        sess = tf.Session()
        output = sess.run(hists)
        sess.close()
        return [np.array(output).tobytes()]
