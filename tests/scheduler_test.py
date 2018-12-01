from scannerpy import Database, DeviceType, Job, Kernel, Config
from scannerpy.stdlib import readers
import scannerpy
import struct
import tempfile
import socket
import requests
import toml
from subprocess import check_call as run

def download_videos():
    # Download video from GCS
    url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        host = socket.gethostname()
        # HACK: special proxy case for Ocean cluster
        if host in ['ocean', 'crissy', 'pismo', 'stinson']:
            resp = requests.get(
                url,
                stream=True,
                proxies={'https': 'http://proxy.pdl.cmu.edu:3128/'})
        else:
            resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        vid1_path = f.name

    # Make a second one shorter than the first
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        vid2_path = f.name
    run([
        'ffmpeg', '-y', '-i', vid1_path, '-ss', '00:00:00', '-t', '00:00:10',
        '-c:v', 'libx264', '-strict', '-2', vid2_path
    ])

    return (vid1_path, vid2_path)

def test_long_pipeline(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    pass_data = db.ops.Pass(input=data)
    pass_data = db.ops.Pass(input=pass_data)
    pass_data = db.ops.Pass(input=pass_data)
    pass_data = db.ops.Pass(input=pass_data)
    pass_data = db.ops.Pass(input=pass_data)
    output_op = db.sinks.Column(columns={'integer': pass_data})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_long_pipeline.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_long_pipeline_wider_than_packet(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    pass_data = db.ops.Pass(input=data)
    pass_data = db.ops.Pass(input=pass_data)
    pass_data = db.ops.Pass(input=pass_data)
    pass_data = db.ops.Pass(input=pass_data)
    pass_data = db.ops.Pass(input=pass_data)
    output_op = db.sinks.Column(columns={'integer': pass_data})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_long_pipeline_wider_than_packet.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

# @scannerpy.register_python_op()
# class PassSecond(Kernel):
#     def __init__(self, config):
#         self.protobufs = config.protobufs

#     def close(self):
#         pass

#     def new_stream(self, args):
#         pass

#     def execute(self, data1: bytes, data2: bytes) -> bytes:
#         return data2

def test_diamond(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    middle_left = db.ops.Pass(input=data)
    middle_right = db.ops.Pass(input=data)
    merge = db.ops.PassSecond(data1=middle_left, data2=middle_right)
    output_op = db.sinks.Column(columns={'integer': merge})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_diamond.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_diamond_wider_than_packet(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    middle_left = db.ops.Pass(input=data)
    middle_right = db.ops.Pass(input=data)
    merge = db.ops.PassSecond(data1=middle_left, data2=middle_right)
    output_op = db.sinks.Column(columns={'integer': merge})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_diamond_wider_than_packet.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_multiple_outputs(db):
    frame = db.sources.FrameColumn()
    index = db.sources.Column()
    output_op = db.sinks.Column(columns={'frame': frame, 'index': index})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        index: db.table('test1').column('index'),
        output_op: 'test_multiple_outputs',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_multiple_outputs.trace')

    num_rows = 0
    for _ in tables[0].column('frame').load():
        num_rows += 1
    assert num_rows == db.table('test1').num_rows()

def test_save_mp4(db):
    frame = db.sources.FrameColumn()
    output = db.sinks.Column(columns={'frame': frame})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output: 'test_save_mp4',
    })
    [out_table] = db.run(output=output, jobs=[job], work_packet_size=8, io_packet_size=64, force=True)
    out_table.profiler().write_trace('test_save_mp4.trace')

    print('Writing output video...')
    out_table.column('frame').save_mp4('{:s}_out'.format("test1"))

def test_triangle(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    middle = db.ops.Pass(input=data)
    merge = db.ops.PassSecond(data1=data, data2=middle)
    output_op = db.sinks.Column(columns={'integer': merge})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_triangle.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_triangle_wider_than_packet(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    middle = db.ops.Pass(input=data)
    merge = db.ops.PassSecond(data1=data, data2=middle)
    output_op = db.sinks.Column(columns={'integer': merge})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_triangle_wider_than_packet.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_triangle_multiple_outputs(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    middle = db.ops.Pass(input=data)
    # merge = db.ops.PassSecond(data1=data, data2=middle)
    output_op = db.sinks.Column(columns={'integer_0': data, 'integer_1': middle})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_triangle_multiple_outputs.trace')

    num_rows = 0
    for buf in tables[0].column('integer_0').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

    num_rows = 0
    for buf in tables[0].column('integer_1').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_triangle_multiple_outputs_wider_than_packet(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    middle = db.ops.Pass(input=data)
    # merge = db.ops.PassSecond(data1=data, data2=middle)
    output_op = db.sinks.Column(columns={'integer_0': data, 'integer_1': middle})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_triangle_multiple_outputs_wider_than_packet.trace')

    num_rows = 0
    for buf in tables[0].column('integer_0').load():
        (val, ) = struct.unpack('=Q', buf)
        print(str(val) + " | " + str(num_rows))
        # assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

    num_rows = 0
    for buf in tables[0].column('integer_1').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_slice(db):
    frame = db.sources.FrameColumn()
    slice_frame = db.streams.Slice(frame, db.partitioner.all(50))
    unsliced_frame = db.streams.Unslice(slice_frame)
    output_op = db.sinks.Column(columns={'frame': unsliced_frame})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_slicing',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_slice.trace')

    num_rows = 0
    for _ in tables[0].column('frame').load():
        num_rows += 1
    assert num_rows == db.table('test1').num_rows()

def test_sample(db):
    def run_sampler_job(sampler, sampler_args, expected_rows):
        frame = db.sources.FrameColumn()
        sample_frame = sampler(input=frame)
        output_op = db.sinks.Column(columns={'frame': sample_frame})

        job = Job(
            op_args={
                frame: db.table('test1').column('frame'),
                sample_frame: sampler_args,
                output_op: 'test_sample',
            })

        tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=200, work_packet_size=200)
        num_rows = 0
        for _ in tables[0].column('frame').load():
            num_rows += 1
        assert num_rows == expected_rows
        tables[0].profiler().write_trace('test.trace')

    # Stride
    expected = int((db.table('test1').num_rows() + 8 - 1) / 8)
    run_sampler_job(db.streams.Stride, {'stride': 8}, expected)
    # Range
    run_sampler_job(db.streams.Range, {'start': 0, 'end': 30}, 30)
    # Strided Range
    run_sampler_job(db.streams.StridedRange, {
        'start': 0,
        'end': 300,
        'stride': 10
    }, 30)
    # Gather
    run_sampler_job(db.streams.Gather, {'rows': [0, 150, 377, 500]}, 4)
    # Gather without selecting row 0
    run_sampler_job(db.streams.Gather, {'rows': [1, 150, 377, 500]}, 4)

def test_space(db):
    def run_spacer_job(spacer, spacing):
        frame = db.sources.FrameColumn()
        hist = db.ops.Histogram(frame=frame)
        space_hist = spacer(input=hist)
        output_op = db.sinks.Column(columns={'histogram': space_hist})

        job = Job(
            op_args={
                frame: db.table('test1').column('frame'),
                space_hist: {
                    'spacing': spacing
                },
                output_op: 'test_space',
            })

        tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=200, work_packet_size=200)
        tables[0].profiler().write_trace('test_space.trace')
        return tables[0]

    # Repeat
    spacing_distance = 8
    table = run_spacer_job(db.streams.Repeat, spacing_distance)
    num_rows = 0
    for hist in table.column('histogram').load(readers.histograms):
        # Verify outputs are repeated correctly
        if num_rows % spacing_distance == 0:
            ref_hist = hist
        assert len(hist) == 3
        for c in range(len(hist)):
            assert (ref_hist[c] == hist[c]).all()
        num_rows += 1
    print("Repeat num_rows: " + str(num_rows))
    assert num_rows == db.table('test1').num_rows() * spacing_distance

    # Null
    table = run_spacer_job(db.streams.RepeatNull, spacing_distance)
    num_rows = 0
    for hist in table.column('histogram').load(readers.histograms):
        # Verify outputs are None for null rows
        if num_rows % spacing_distance == 0:
            assert hist is not None
            assert len(hist) == 3
            assert hist[0].shape[0] == 16
        else:
            assert hist is None
        num_rows += 1
    print("Null num_rows: " + str(num_rows))
    assert num_rows == db.table('test1').num_rows() * spacing_distance

def test_overlapping_slice(db):
    frame = db.sources.FrameColumn()
    slice_frame = db.streams.Slice(frame)
    sample_frame = db.streams.Range(slice_frame)
    unsliced_frame = db.streams.Unslice(sample_frame)
    output_op = db.sinks.Column(columns={'frame': unsliced_frame})
    job = Job(
        op_args={
            frame:
            db.table('test1').column('frame'),
            slice_frame:
            db.partitioner.strided_ranges([(0, 15), (5, 25), (15, 35)], 1),
            sample_frame: [
                (0, 10),
                (5, 15),
                (5, 15),
            ],
            output_op:
            'test_slicing',
        })

    tables = db.run(output_op, [job], force=True, show_progress=False)

    num_rows = 0
    for _ in tables[0].column('frame').load():
        num_rows += 1
    assert num_rows == 30

def test_load_video_column(db):
    for name in ['test1', 'test1_inplace']:
        next(db.table(name).load(['frame']))

def test_gather_video_column(db):
    for name in ['test1', 'test1_inplace']:
        # Gather rows
        rows = [0, 10, 100, 200]
        frames = [_ for _ in db.table(name).load(['frame'], rows=rows)]
        assert len(frames) == len(rows)

def test_packed_file_source(db):
    # Write test file
    path = '/tmp/cpp_source_test'
    with open(path, 'wb') as f:
        num_elements = 4
        f.write(struct.pack('=Q', num_elements))
        # Write sizes
        for i in range(num_elements):
            f.write(struct.pack('=Q', 8))
        # Write data
        for i in range(num_elements):
            f.write(struct.pack('=Q', i))

    data = db.sources.PackedFile()
    pass_data = db.ops.Pass(input=data)
    output_op = db.sinks.Column(columns={'integer': pass_data})
    job = Job(op_args={
        data: {
            'path': path
        },
        output_op: 'test_cpp_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_packed_file_source.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()

def test_bounded_state(db):
    warmup = 3

    frame = db.sources.FrameColumn()
    increment = db.ops.TestIncrementBounded(ignore=frame, bounded_state=warmup)
    sampled_increment = db.streams.Gather(increment, [0, 10, 25, 26, 27])
    output_op = db.sinks.Column(columns={'integer': sampled_increment})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_slicing',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)
    tables[0].profiler().write_trace('test_bounded_state.trace')

    num_rows = 0
    expected_output = [0, warmup, warmup, warmup + 1, warmup + 2]
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=q', buf)
        assert val == expected_output[num_rows]
        print(num_rows)
        num_rows += 1
    assert num_rows == 5

def test_bounded_state_wider_than_packet(db):
    warmup = 3

    frame = db.sources.FrameColumn()
    increment = db.ops.TestIncrementBounded(ignore=frame, bounded_state=warmup)
    sampled_increment = db.streams.Gather(increment, [0, 10, 25, 26, 27])
    output_op = db.sinks.Column(columns={'integer': sampled_increment})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_slicing',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_bounded_state_wider_than_packet.trace')

    num_rows = 0
    expected_output = [0, warmup, warmup, warmup + 1, warmup + 2]
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=q', buf)
        assert val == expected_output[num_rows]
        print(num_rows)
        num_rows += 1
    assert num_rows == 5


def test_unbounded_state(db):
    frame = db.sources.FrameColumn()
    slice_frame = db.streams.Slice(frame, db.partitioner.all(50))
    increment = db.ops.TestIncrementUnbounded(ignore=slice_frame)
    unsliced_increment = db.streams.Unslice(increment)
    output_op = db.sinks.Column(columns={'integer': unsliced_increment})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_slicing',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)

    num_rows = 0
    for _ in tables[0].column('integer').load():
        num_rows += 1
    assert num_rows == db.table('test1').num_rows()

def test_files_source(db):
    # Write test files
    path_template = '/tmp/files_source_test_{:d}'
    num_elements = 4
    paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        with open(path, 'wb') as f:
            # Write data
            f.write(struct.pack('=Q', i))
        paths.append(path)

    data = db.sources.Files()
    pass_data = db.ops.Pass(input=data)
    output_op = db.sinks.Column(columns={'integer': pass_data})
    job = Job(op_args={
        data: {
            'paths': paths
        },
        output_op: 'test_files_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    tables[0].profiler().write_trace('test_files_source.trace')

    num_rows = 0
    for buf in tables[0].column('integer').load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == tables[0].num_rows()


def test_files_sink(db):
    # Write initial test files
    path_template = '/tmp/files_source_test_{:d}'
    num_elements = 4
    input_paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        with open(path, 'wb') as f:
            # Write data
            f.write(struct.pack('=Q', i))
        input_paths.append(path)

    # Write output test files
    path_template = '/tmp/files_sink_test_{:d}'
    num_elements = 4
    output_paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        output_paths.append(path)

    data = db.sources.Files()
    pass_data = db.ops.Pass(input=data)
    output_op = db.sinks.Files(input=pass_data)
    job = Job(op_args={
        data: {
            'paths': input_paths
        },
        output_op: {
            'paths': output_paths
        }
    })

    tables = db.run(output_op, [job], force=True, show_progress=False, io_packet_size=1, work_packet_size=1)
    # tables[0].profiler().write_trace('test_files_sink.trace')

    # Read output test files
    for i in range(num_elements):
        path = output_paths[i]
        with open(path, 'rb') as f:
            # Write data
            d, = struct.unpack('=Q', f.read())
            assert d == i

def test_stencil(db):
    frame = db.sources.FrameColumn()
    sample_frame = db.streams.Range(frame, 0, 1)
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[-1, 0])
    output_op = db.sinks.Column(columns={'flow': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil1.trace')

    num_rows = 0
    for _ in tables[0].column('flow').load():
        num_rows += 1
    assert num_rows == 1

    frame = db.sources.FrameColumn()
    sample_frame = db.streams.Range(frame, 0, 1)
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output_op = db.sinks.Column(columns={'flow': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil2.trace')

    frame = db.sources.FrameColumn()
    sample_frame = db.streams.Range(frame, 0, 2)
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output_op = db.sinks.Column(columns={'flow': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil3.trace')

    num_rows = 0
    for _ in tables[0].column('flow').load():
        num_rows += 1
    assert num_rows == 2

    frame = db.sources.FrameColumn()
    flow = db.ops.OpticalFlow(frame=frame, stencil=[-1, 0])
    sample_flow = db.streams.Range(flow, 0, 1)
    output_op = db.sinks.Column(columns={'flow': sample_flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil4.trace')

    num_rows = 0
    for _ in tables[0].column('flow').load():
        num_rows += 1
    assert num_rows == 1

def test_stencil_avg(db):
    frame = db.sources.FrameColumn()
    sample_frame = db.streams.Range(frame, 0, 4)
    flow = db.ops.AverageImage(frame=sample_frame, stencil=[-2, -1, 0, 1, 2])
    output_op = db.sinks.Column(columns={'avg': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil1.trace')

    num_rows = 0
    for _ in tables[0].column('avg').load():
        num_rows += 1
    print(num_rows)

    frame = db.sources.FrameColumn()
    flow = db.ops.AverageImage(frame=frame, stencil=[-2, -1, 0, 1, 2])
    output_op = db.sinks.Column(columns={'avg': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil2.trace')
    for _ in tables[0].column('avg').load():
        num_rows += 1
    print(num_rows)

    frame = db.sources.FrameColumn()
    sample_frame = db.streams.Range(frame, 0, 2)
    flow = db.ops.AverageImage(frame=sample_frame, stencil=[-2, -1, 0, 1, 2])
    output_op = db.sinks.Column(columns={'avg': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil3.trace')

    num_rows = 0
    for _ in tables[0].column('avg').load():
        num_rows += 1
    print(num_rows)

    frame = db.sources.FrameColumn()
    flow = db.ops.AverageImage(frame=frame, stencil=[-2, -1, 0, 1, 2])
    sample_flow = db.streams.Range(flow, 0, 1)
    output_op = db.sinks.Column(columns={'avg': sample_flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_stencil4.trace')

    num_rows = 0
    for _ in tables[0].column('avg').load():
        num_rows += 1
    print(num_rows)

def test_wider_than_packet_stencil(db):
    frame = db.sources.FrameColumn()
    sample_frame = db.streams.Range(frame, 0, 3)
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output_op = db.sinks.Column(columns={'flow': flow})
    job = Job(op_args={
        frame: db.table('test1').column('frame'),
        output_op: 'test_sencil',
    })

    tables = db.run(
        output_op, [job],
        force=True,
        show_progress=False,
        io_packet_size=1,
        work_packet_size=1,
        pipeline_instances_per_node=1)
    tables[0].profiler().write_trace('test_wider_than_packet_stencil.trace')

    num_rows = 0
    for _ in tables[0].column('flow').load():
        num_rows += 1
    assert num_rows == 3

def make_config(master_port=None, worker_port=None, path=None):
    cfg = Config.default_config()
    cfg['network']['master'] = '35.235.81.112'
    cfg['storage']['db_path'] = '/Users/swjz/.scanner/db'
    if master_port is not None:
        cfg['network']['master_port'] = master_port
    if worker_port is not None:
        cfg['network']['worker_port'] = worker_port

    if path is not None:
        with open(path, 'w') as f:
            cfg_path = path
            f.write(toml.dumps(cfg))
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cfg_path = f.name
            f.write(bytes(toml.dumps(cfg), 'utf-8'))
    return (cfg_path, cfg)

db = Database()

# Config for multiple workers
# master = 'localhost:5055'
# workers = ['localhost:{:04d}'.format(5060 + d) for d in range(6)]
# cfg_path = '/Users/swjz/Dropbox/coding/scanner/examples/tutorials/config_local.toml'
# db = Database(config_path=cfg_path,
#               # no_workers_timeout=120,
#               master=master,
#               workers=workers,
#               enable_watchdog=False,
#               debug=False)

print(db.summarize())
(vid1_path, vid2_path) = download_videos()
print(vid1_path)
print(vid2_path)
db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)], force=True)
db.ingest_videos(
    [('test1_inplace', vid1_path), ('test2_inplace', vid2_path)],
    inplace=True, force=True)



# Passed:
test_long_pipeline(db)
print("Done long pipeline test!")
test_long_pipeline_wider_than_packet(db)
print("Done long pipeline wider than packet test!")
test_diamond(db)
print("Done diamond test!")
test_diamond_wider_than_packet(db)
print("Done diamond wider than packet test!")
test_triangle(db)
print("Done triangle test!")
test_triangle_wider_than_packet(db)
print("Done triangle wider than packet test!")
test_load_video_column(db)
print("Done load video column test!")
test_gather_video_column(db)
print("Done gather video column test!")
test_space(db)
print("Done space test!")
test_sample(db)
print("Done sample test!")
test_save_mp4(db)
print("Done save mp4 test!")
test_stencil(db)
print("Done stencil test!")
test_packed_file_source(db)
print("Done packed file source test!")
test_files_source(db)
print("Done files source test!")
test_files_sink(db)
print("Done files sink test!")
test_triangle_multiple_outputs(db)
print("Done triangle multiple outputs test!")
test_bounded_state(db)
print("Done bounded state test!")

# To be fixed:
# test_multiple_outputs(db)
# print("Done multiple outputs test!")
# test_stencil_avg(db)
# print("Done stencil average test!")
# test_wider_than_packet_stencil(db)
# print("Done wider than packet stencil test!")
# test_triangle_multiple_outputs_wider_than_packet(db)
# print("Done triangle multiple outputs wider than packet test!")

# Ignored:
# test_slice(db)
# test_overlapping_slice(db)
# test_unbounded_state(db)
# test_bounded_state_wider_than_packet(db)

