.. _stored-streams:

Stored Streams
==============

Overview
--------
Scanner represents input and output data as sequences of data items called :py:class:`~scannerpy.storage.StoredStream` s. Stored streams are Python objects that describe to Scanner how to read data to be processed and how to write data after it has been processed by a Scanner application. Stored streams can represent data stored in a variety of locations or formats:

- Video files (mp4, mkv, etc).
- Collections of files (images, text, etc)
- Packed binary files (custom RAW formats).
- SQL tables (image metadata)

For example, the following code creates a stored stream for a video file named :code:`example.mp4`:

.. code-block:: python

   from scannerpy import Client
   from scannerpy.storage import NamedVideoStream
   sc = Client()
   video_stream = NamedVideoStream(sc, 'example', path='example.mp4')

:py:class:`~scannerpy.storage.NamedVideoStream` is a special type of stored stream which represents data stored inside Scanner's internal datastore. In order to efficiently read frames from a video, Scanner needs to build an index over the compressed video. By specifying :code:`path = 'example.mp4`, we've told Scanner to initialize a stream named :code:`example` from the :code:`example.mp4` video.

Another example of a stored stream is the :py:class:`~scannerpy.storage.FilesStream`, which represents a stream of individual files stored on a filesystem or cloud blob storage:

.. code-block:: python

   from scannerpy.storage import FilesStream
   image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
   file_stream = FilesStream(sc, paths=image_paths)

By default, :py:class:`~scannerpy.storage.FilesStream` reads from the local filesystem. However, like all stored streams, the storage location and configuration options can be specified with :py:class:`~scannerpy.storage.StorageBackend` s.

Storage Backends 
----------------
:py:class:`~scannerpy.storage.StorageBackend` s represent the specific storage location or format for stored streams. For example, :py:class:`~scannerpy.storage.FilesStream` can be configured to read files from Amazon's S3 storage service instead of the default local file system by creating the appropriate storage backend:

.. code-block:: python

   from scannerpy.storage import FileStroage, FilesStream
   image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
   file_storage = FileStorage(storage_type='s3',
                              bucket='example-bucket',
                              region='us-west-1')
   file_stream = FilesStream(sc, paths=image_paths, storage=file_storage)

I/O Operations
--------------
I/O operations allow Scanner applications to read and write to stored streams. To read from a stored stream, Scanner applications construct :py:meth:`~scannerpy.io.IOGenerator.Input` operations, specifying a list of stored streams:

.. code-block:: python

   frame = sc.io.Input([video_stream])

This code creates a sequence of video frames, :code:`frame`, that can be used in the context of a Scanner computation graph to read the video specified by :code:`video_stream` (to learn more about computation graphs, check out the :ref:`graphs`. guide). Stored streams are also used to specify where to write data to:

.. code-block:: python

   output_video_stream = NamedVideoStream(sc, 'example-output')
   frame = sc.io.Output(frame, [output_video_stream])

Here, the frames we read in from before will be written back out to a :py:class:`~scannerpy.storage.NamedVideoStream` called :code:`example-output`.

Reading Data Locally
--------------------
Stored streams can be read directly in Python by calling the :py:meth:`~scannerpy.storage.StoredStream.load` method:

.. code-block:: python

   for frame in video_stream.load():
       print(frame.shape)

Reading from this stream lazily loads video frames from :code:`video_stream` as numpy arrays. If we were reading bounding boxes or some other data format, the :code:`load` method would return data elements formatted according to the data type of the stream.

Deleting Stored Streams
-----------------------
Stored stream data is persistent: unless a stored stream is explicitly deleted, the data will stay around and can be used in future Scanner applications. A stored stream can be deleted by invoking the :py:meth:`~scannerpy.storage.StoredStream.delete` method:

.. code-block:: python

   video_stream.delete(sc)

If there are multiple streams to delete, it can be more efficient to invoke a bulk delete operation by calling :py:meth:`~scannerpy.storage.StorageBackend.delete` on the storage backend itself:

.. code-block:: python

   video_stream.storage().delete(sc, [...])

..
    - Introduce what stored streams are used  for in scanner
    - Talk about storage objects
    - Give an example/  show syntax
    - Talk about how they are used in scanner graphs
    - Explain how multiple streams can be used in a scanner graph
    - Persistence of streams
    - Explain API for streams/storage objects
