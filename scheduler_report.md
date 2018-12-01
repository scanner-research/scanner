## Scheduler Functionality Report

### What it supports

- Any kind of DAG
- Stencil op
- Variable task size for each op except for built-in ops and sink op
- Multiple workers on single machine
- Multiple machines with CPU only

- Fault tolerance: blacklist job, reassign task to worker
- Single GPU

### What it doesn't support for now

- Multiple GPU devices

- Everything related to Slice

  - Slice and overlapping slice
  - Bounded state (with a `Gather` sampler wider than task size)
  - Unbounded state test

- Multi-thread workers

- Variable task size for built-in ops and sink op

- Variable task size for source op has bugs:

  - When a load (and pre-evaluate) task only processes part of the source, it might cause some problem. That's because in current implementation, the enumerators are not split into task granularity. For one video source, there is only one enumerator even if the video has 10,000 frames and the task size is 10. So my workaround was to hack `load_worker.cpp`: use `row_start` and `row_end` to mark row offsets for this particular task. However, this still causes some problems: if the source op is followed by a gather op and row 0 is not selected by gather op, `row_start` will be the first row_id selected by gather op; but actually `row_start` should be 0 because `source_args.row_ids` has already excluded the rows not selected by gather. This causes failing of "Gather without selecting row 0" in `test_sample()`.

- Old liveness analysis (`dead_columns` and `unused_outputs`) does not work, but both stencil and `db.streams.Range` seem to need it:

  Our current workaround is to look for how many output columns are needed, and only give the last one or two columns to post-evaluate. We also merge multiple columns from multiple tasks to one column if these tasks come from the same op. However, when stencil op or `db.stream.Range` grabs necessary rows from multiple columns (multiple source tasks), it does not merge them together to be one single column. Since the post-evaluate is not aware of the stencil status, some rows from former columns are missing. This causes failing of `test_stencil_avg()` and `test_wider_than_packet_stencil()`.

- The algorithm for merging tasks has bugs with variable task size for source op:

  We currently merge them only if the source task does not belong to a source op. If the source task belongs to a source op, it might have multiple input columns because we remapped input columns, so we don't merge them. However, if we split source op into multiple tasks and have a sink op immediately after source op, they also don't get merged at all. This causes failing of `test_wider_than_packet_stencil()` and `test_triangle_multiple_outputs_wider_than_packet()`.

### Working tests in [`scheduler_test.py`](https://github.com/swjz/scanner/blob/scheduler/tests/scheduler_test.py)

```
test_long_pipeline()
test_long_pipeline_wider_than_packet()  // "packet" here means task size
test_diamond()
test_diamond_wider_than_packet()
test_triangle()
test_triangle_wider_than_packet()
test_load_video_column()
test_gather_video_column()
test_space()
test_save_mp4()
test_stencil()
test_packed_file_source()
test_files_source()
test_files_sink()
test_triangle_multiple_outputs()
test_bounded_state()
```

### Failing tests in [`scheduler_test.py`](https://github.com/swjz/scanner/blob/scheduler/tests/scheduler_test.py)

```
test_sample()  // only "Gather without selecting row 0" fails
test_multiple_outputs()
test_stencil_avg()
test_wider_than_packet_stencil()
test_triangle_multiple_outputs_wider_than_packet()
```

### How to specify task size

Currently, if you set `io_packet_size` to a fixed number, the task size for all ops (excluding built-in ops and sink op) will be set to this number. Related code: https://github.com/swjz/scanner/blob/9d780f6c3ea40b65b6b70f89a4fb57d1c7279f8a/scanner/engine/dag_analysis.cpp#L1308-L1316, https://github.com/swjz/scanner/blob/9d780f6c3ea40b65b6b70f89a4fb57d1c7279f8a/scanner/engine/worker.cpp#L1954-L1957, https://github.com/swjz/scanner/blob/9d780f6c3ea40b65b6b70f89a4fb57d1c7279f8a/scanner/engine/master.cpp#L1658-L1663.



### TODO

1. Fuse tasks

2. Multiple threads within one worker, like Alex's scheduler

   1. Pre/Post running at the same time as Evaluate, within one worker

   2. Load/Save running at the same time as Evaluate, within one worker

      I/O and evaluate at the same time

3. CPU work and GPU work running at the same time