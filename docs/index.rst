.. scanner documentation master file, created by
   sphinx-quickstart on Sun Nov 26 19:06:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================================

.. raw:: html

  <div class="row">
    <div class="col-xs-12">
      <div class="text-center">
        <h2>
          Scanner is an open-source distributed system for authoring video processing applications.
        </h2>
        <a href="/guide.html" class="btn btn-default btn-sm">
           Get started
        </a>
        <a href="/overview.html" class="btn btn-default btn-sm">
           How does it work?
        </a>
      </div>
    </div>
  </div>

  <div class="row">
    <div class="col-md-6 col-xs-12">
      <h3>
       Simpler video processing
      </h3>
      <ul>
        <li>Out-of-the box modules for object, pose, and face detection.</li>
        <li>Works directly with video files (no more pre-processing into images!).</li>
        <li>Supports efficent random access of frames in video.</li>
        <li>Scales to use multiple GPUs and multi-core CPUs.</li>
        <li>Scales out to hundreds of machines in the cloud or your local cluster.</li>
        <li>Integrated with Kubernetes, Google Cloud Platform, and AWS.</li>
      </p>
    </div>

    <div class="col-md-6 col-xs-12">
      <h3>
       Easy to use Python API
      </h3>
         <pre><code class="jljs python">from scannerpy import Database, Job
  
  # Ingest a video 
  db = Database()
  db.ingest_videos([('example_table', 'example.mp4')])
  
  # Define a Computation Graph
  frame = db.sources.FrameColumn()                                    
  hist = db.ops.Histogram(input=frame)            
  output_frame = db.sinks.Column(columns={'hist': hist})         
  
  # Set parameters of computation graph ops
  job = Job(op_args={
    # Column to read input frames from
    frame: db.table('example_table').column('frame'),
    # Table name for computation output
    output_frame: 'resized_example'                 
  })
  
  # Execute the computation graph 
  output_tables = db.run(output=output_frame, jobs=[job])
         </code></pre>
    </div>
  </div>

  <div class="row">
    <div class="col-xs-12">
      <h3>
      Applications
      </h3>
    </div>

    <div class="col-md-6 col-xs-12">
      <h4>
       Analyzing videos with machine learning and computer vision
      </h4>
      <p>
        Use modern neural networks to automatically detect objects, people, and faces in your videos.
        Check out <a href="https://scanner-research.github.io/scannertools/">scannertools</a> for an
        easy to use toolkit that's setup to download and run these models.
      </p>
      <div style="position: relative; padding-bottom: 56.25%; padding-top: 25px; height: 0;">
        <iframe style="position: absolute; top: 0; left: 0;" width="100%" height="100%" src="https://www.youtube.com/embed/IQsb_nbPf9M" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
      </div>
    </div>

    <div class="col-md-6 col-xs-12">
      <h4>
        VR video synthesis from multiple cameras
      </h4>
      <p>
        Scanner is currently being used as the compute engine behind the <a href="https://facebook360.fb.com/2018/09/26/film-the-future-with-red-and-facebook-360/">Manifold</a> 360 video camera from Facebook and RED. Scanner has been integrated into the publicly available version of the Surround 360 system on <a href="https://github.com/scanner-research/Surround360">GitHub</a>.
      </p>
      <div style="position: relative; padding-bottom: 56.25%; padding-top: 25px; height: 0;">
        <iframe style="position: absolute; top: 0; left: 0;" src="https://www.facebook.com/plugins/video.php?href=https%3A%2F%2Fwww.facebook.com%2FFacebook360%2Fvideos%2F370097893531240%2F%3Ft%3D0&&show_text=false" width="100%" height="100%" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowTransparency="true" allow="encrypted-media" allowFullScreen="true"></iframe>
      </div>
    </div>
  </div>
