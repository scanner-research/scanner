.. _scannertools-docs:

Scannertools API
================

.. raw:: html
         <h1>scannertools API</h1>
============

.. toctree::
   :maxdepth: 1

Operations
----------

Face Detection
~~~~~~~~~~~~~~


.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/IQsb_nbPf9M" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. py:function:: ops.MTCNNDetectFaces(frame)

   Detect human poses in an image.
   Registered with :code:`import scannertools.face_detection`.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frames to detect poses in.
   :return: A list of bounding boxes.
   :rtype: scannerpy.types.BboxList

.. code-block:: python

   import scannerpy as sp
   import scannertools.face_detection

   def main():
     # Compute face bounding boxes and draw them on a sample video
     with sp.utils.sample_video() as video_path:
       cl = sp.Client()
       video = sp.NamedVideoStream(cl, 'example', path=video_path)
       frames = cl.io.Input([video])
       faces = cl.ops.MTCNNFaceDetect(frame=frames)
       drawn_faces = cl.ops.DrawBboxes(frame=frames, bboxes=faces)
       output_video = sp.NamedVideoStream(cl, 'example_faces')
       output_op = cl.io.Output(drawn_faces, [output_video])
       cl.run(output_op, sp.PerfParams.estimate())
       output_video.save_mp4('example_faces')
       # output video is saved to 'example_faces.mp4'

   if __name__ == "__main__":
       main()

Face Embedding
~~~~~~~~~~~~~~

.. py:function:: ops.EmbedFaces(frame, bboxes)

   Compute a face embeddings vector for each bounding box in the image.
   Registered with :code:`import scannertools.face_embedding`.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frames to detect poses in.
   :param scannerpy.types.BboxList bboxes: Bounding boxes to compute face embeddings on.
   :return: Serialized face embeddings.
   :rtype: bytes

Gender Detection
~~~~~~~~~~~~~~~~

.. py:function:: ops.DetectGender(frame, bboxes)

   Detect the gender of people in an image.
   Registered with :code:`import scannertools.gender_detection`.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frames to detect gender in.
   :param scannerpy.types.BboxList bboxes: Bounding boxes to indicate where to estimate gender.
   :return: The detected genders.
   :rtype: Any

Object Detection
~~~~~~~~~~~~~~~~

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/6xt-YVFCC9I" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. py:function:: ops.DetectObjects(frame, bboxes)

   Detect objects in an image.
   Registered with :code:`import scannertools.object_detection`.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frames to detect objects in.
   :return: A list of bounding boxes for the detected objects.
   :rtype: scannerpy.types.BboxList


Pose Detection
~~~~~~~~~~~~~~

.. raw:: html

         <iframe width="560" height="315" src="https://www.youtube.com/embed/N1bT1yjnvMY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. py:function:: ops.OpenPose(model_directory, pose_num_scales, pose_scale_gap, compute_hands, hand_num_scales, hand_scale_gap, compute_face, frame)

   Detect human poses in an image.
   Registered with :code:`import scannertools_caffe`.

   Init parameters:

   :param str model_directory: A path to the directory with the OpenPose model files.
   :param int pose_num_scales: The number of scales to evaluate the pose network on.
   :param float pose_scales_gap: The scaling factor between scales for the pose network.
   :param bool compute_hands: Flag to enable computing hand keypoints.
   :param int hand_num_scales: The number of scales to evaluate the hand network on.
   :param float hand_scale_gap: The scaling factor between scales for the hand network.
   :param bool compute_face: Flag to enable computing face keypoints.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frames to detect poses in.
   :return: A list of detected poses.
   :rtype: scannertools_caffe.pose_detection.PoseList

Shot Detection
~~~~~~~~~~~~~~

.. image:: https://storage.googleapis.com/scanner-data/public/sample-shots-small.jpg
   :target: https://storage.googleapis.com/scanner-data/public/sample-shots.jpg
   :scale: 50%

.. py:function:: ops.ShotBoundaries(histograms)

   Detect shot boundaries using color histograms of frames.
   Registered with :code:`import scannertools.shot_detection`.

   Stream parameters:

   :param scannerpy.types.Histogram histograms: The color histograms of frames from a video.
   :return: The indices of the shot boundaries.
   :rtype: Sequence[int]

Resize
~~~~~~

.. py:function:: ops.Resize(width, height, min, preserve_aspect, interpolation, frame)

   Resize images to a fixed size.
   Registered with :code:`import scannertools.imgproc`.

   Stream config parameters:

   :param Sequence[int] width: The target width of the frame.
   :param Sequence[int] height: The target height of the frame.
   :param Sequence[bool] min: If true, resizes frames to :code:`width` and :code:`height` only if the input frames width and height if both are less than :code:`width` and :code:`height`.
   :param Sequence[bool] preserve_aspect: If true, sets :code:`width` to preserve the aspect ratio of the input frame if :code:`height` is non-zero. Likewise if :code:`width` is non-zero.
   :param Sequence[str] interpolation: The type of interpolation to use.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frame to resize.
   :return: The resized frame.
   :rtype: scannerpy.types.FrameType

Optical Flow
~~~~~~~~~~~~

.. py:function:: ops.OpticalFlow(frame)

   Computes optical flow from one frame to the next.
   Registered with :code:`import scannertools.imgproc`.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frame to compute flow on.
   :return: The flow field.
   :rtype: scannerpy.types.FrameType

Histogram
~~~~~~~~~

.. py:function:: ops.Histogram(frame)

   Computes a color histogram of the frame.
   Registered with :code:`import scannertools.imgproc`.

   Stream parameters:

   :param scannerpy.types.FrameType frame: The frame to process.
   :return: The color histogram.
   :rtype: scannerpy.types.Histogram

Tracker
~~~~~~~

.. automodule:: scannertools.tracker
    :members:
    :undoc-members:
    :show-inheritance:

Utilities for writing Operations
--------------------------------

Caffe2
~~~~~~

.. automodule:: scannertools.caffe
    :members:
    :undoc-members:
    :show-inheritance:

Pytorch
~~~~~~~

.. automodule:: scannertools.torch
    :members:
    :undoc-members:
    :show-inheritance:

TensorFlow
~~~~~~~~~~

.. automodule:: scannertools.tensorflow
    :members:
    :undoc-members:
    :show-inheritance:

Storage
-------

.. automodule:: scannertools.storage.files_storage
    :members:
    :undoc-members:
    :show-inheritance:

scannertools_caffe
------------------

.. automodule:: scannertools_caffe.pose_detection
    :members:
    :undoc-members:
    :show-inheritance:
