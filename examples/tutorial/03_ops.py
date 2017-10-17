from scannerpy import Database, Job, DeviceType

################################################################################
# This tutorial shows how to combine multiple operators into a computation     #
# graph and wire inputs/outputs.                                               #
################################################################################

with Database() as db:

    # Scanner can take a directed acyclic graph (DAG) of operators and pass data
    # between them. Each graph has starts with data from an input table.
    frame = db.ops.FrameInput()

    blurred_frame = db.ops.Blur(
        frame = frame,
        kernel_size = 3,
        sigma = 0.5)

    # Multiple operators can be hooked up in a computation by using the outputs
    # of one as the inputs of another.
    histogram = db.ops.Histogram(
        frame = blurred_frame)

    output_op = db.ops.Output(columns=[histogram])

    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            output_op: 'output_table',
        }
    )
    bulk_job = BulkJob(dag=dag, jobs=[job])

    db.run(bulk_job, force=True)

    # Ops can have several attributes that affect which stream elements they
    # will receive or how they will receive them. These attributes include:
    #
    # - Batch: The Op can receive multiple elements at once to enable SIMD
    #          or vector-style processing.
    #
    # - Stencil: The Op requires a window of input elements (for example, the
    #            previous and next element) at the same time to produce an
    #            output.
    #
    # - Bounded State: For each output, the Op requires at least W sequential
    #                  "warmup" elements before it can produce a valid output.
    #                  For example, if the output of this Op is sampled
    #                  sparsely, this guarantees that the Op can "warmup"
    #                  its state on a stream of W elements before producing the
    #                  requested output.
    #
    # - Unbounded State: This Op will always process all preceding elements of
    #                    its input streams before producing a requested output.
    #                    This means that sampling operations after this Op
    #                    can not change how many inputs it receives. In the next
    #                    tutorial, we will show how this can be relaxed for
    #                    sub-streams of the input.
    #
    # The rest of this tutorial will show examples of each attribute in action.


    # Batch
    # Here we specify that the histogram kernel should receive a batch of 8
    # input elements at once. Logically, each element is still processed
    # independently but multiple elements are provided for efficient
    # batch processing. If there are not enough elements left in a stream,
    # the Op may receive less than a batch worth of elements.
    histogram = db.ops.Histogram(
        frame = frame,
        batch = 8)


    # Stencil
    # Under construction...
    diff = db.ops.FrameDifference(
        frame = frame,
        stencil = [-1, 0])


    # Bounded State
    # Under construction... *digging man GIF*

    # Unbounded State
    # Under construction... *spinning construction sign GIF*
