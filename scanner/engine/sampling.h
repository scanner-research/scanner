  struct GroupSample {
    i32 group_index;
    std::vector<i32> frames;
  };
  std::vector<GroupSample> sampled_frames() const;

  struct RowLocations {
    // For regular columns
    std::vector<i32> work_items;
    std::vector<Interval> work_item_intervals;
  };
  // Gets the list of work items for a sequence of rows in the job
  RowLocations row_work_item_locations(Sampling sampling, i32 group_id,
                                       const LoadWorkEntry& entry) const;

  struct FrameLocations {
    // For frame column
    std::vector<Interval> intervals;
    std::vector<DecodeArgs> video_args;
    std::vector<ImageDecodeArgs> image_args;
  };
  FrameLocations frame_locations(Sampling sampling, i32 group_index,
                                 const LoadWorkEntry& entry) const;
