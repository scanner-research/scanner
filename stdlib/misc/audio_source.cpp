#include "scanner/api/enumerator.h"
#include "scanner/api/source.h"
#include "scanner/util/ffmpeg.h"
#include "stdlib/stdlib.pb.h"

#include <math.h>
#include <fstream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
}

using storehouse::StorageBackend;

// TODO(wcrichto): check whole impl for memory leaks

namespace scanner {

#define FF_ERROR(EXPR)                \
  {                                   \
    i32 err = EXPR;                   \
    char errbuf[1024];                \
    av_strerror(err, errbuf, 1024);   \
    LOG_IF(FATAL, err < 0) << errbuf; \
  }

class AudioDecoder {
 public:
  AudioDecoder(std::string& path, std::shared_ptr<StorageBackend> storage, Profiler* profiler)
    : path_(path), storage_(storage), profiler_(profiler) {
    ProfileBlock _block(profiler_, "decode_setup");

    // https://stackoverflow.com/questions/39778605/ffmpeg-avformat-open-input-not-working-invalid-data-found-when-processing-input
    av_register_all();

    // Initialize ffmpeg reader object
    std::string error_message;
    if (!ffmpeg_storehouse_state_init(&file_state_, storage_.get(), path_,
                                      error_message)) {
      LOG(FATAL) << error_message;
    }

    // Initialize format context for reading from video stream
    size_t avio_context_buffer_size = 4096;
    avio_context_buffer_ =
        static_cast<u8*>(av_malloc(avio_context_buffer_size));
    format_context_ = avformat_alloc_context();
    io_context_ = avio_alloc_context(
        avio_context_buffer_, avio_context_buffer_size, 0, &file_state_,
        &ffmpeg_storehouse_read_packet, NULL, &ffmpeg_storehouse_seek);
    format_context_->pb = io_context_;

    // Open I/O connection to video
    FF_ERROR(avformat_open_input(&format_context_, NULL, NULL, NULL));

    // Get metadata
    FF_ERROR(avformat_find_stream_info(format_context_, NULL));

    // Find index of audio stream
    i32 stream_index = av_find_best_stream(format_context_, AVMEDIA_TYPE_AUDIO,
                                           -1, -1, &codec_, 0);
    LOG_IF(FATAL, stream_index < 0) << "could not find best stream";
    stream_ = format_context_->streams[stream_index];
    time_base_ = av_q2d(stream_->time_base);
    VLOG(1) << "stream index: " << stream_index;

    // Set all other streams to discard their packets
    for (i32 i = 0; i < format_context_->nb_streams; ++i) {
      if (i != stream_index) {
        format_context_->streams[i]->discard = AVDISCARD_ALL;
      }
    }

    // Prepare the codec context for decoding
    context_ = stream_->codec;
    FF_ERROR(avcodec_open2(context_, codec_, NULL));

    // Initialize data packet (raw encoded audio bytes)
    av_init_packet(&packet_);
  }

  ~AudioDecoder() {
    avcodec_free_context(&context_);

    // TODO(wcrichto): this causes a double free?
    // avformat_close_input(&format_context_);

    av_freep(&io_context_->buffer);
    av_freep(&io_context_);
  }

  double duration() {
    LOG_IF(FATAL, stream_->duration == AV_NOPTS_VALUE)
        << "Duration is not set";
    return stream_->duration * time_base_;
  }

  void decode(const std::vector<i64>& rows, const f64 target_frame_size_sec,
             std::vector<Frame*>& frames) {
    ProfileBlock _block(profiler_, "decode");

    i32 sample_rate = context_->sample_rate;
    i32 source_frame_size_samples = context_->frame_size;
    i32 target_frame_size_samples = target_frame_size_sec * sample_rate;

    f32* frames_block = (f32*)new_block_buffer(
        CPU_DEVICE, target_frame_size_samples * rows.size() * sizeof(f32),
        rows.size());
    f32* cur_frame = frames_block;

    for (i32 row_idx = 0; row_idx < rows.size(); ++row_idx) {
      f64 time = rows[row_idx] * target_frame_size_sec;

      VLOG(1) << "NEW TIME: " << time;

      // Seeking to first avframe containing the first target sample if previous
      // decode was non-contiguous
      // if (row_idx == 0 || rows[row_idx] != rows[row_idx - 1] + 1) {
      seek(time);
      //}

      std::vector<AVFrame*> av_frames;
      i32 cur_av_frame_idx = 0;
      i32 samples_so_far = 0;

      // Special case the first frame, since we aren't reading from first sample
      // in packet. Figure out which sample corresponds to the first desired
      // sample at the provided time
      decode_packet(av_frames);
      AVFrame* av_frame = av_frames[cur_av_frame_idx];
      VLOG(1) << "time: "
              << (av_frame_get_best_effort_timestamp(av_frame) * time_base_);
      i64 sample_offset =
          (time - (av_frame_get_best_effort_timestamp(av_frame) * time_base_)) *
          sample_rate;
      LOG_IF(FATAL, sample_offset >= source_frame_size_samples)
          << "sample_offset was bigger than source frame size";

      VLOG(1) << "Sample offset: " << sample_offset;

      // Read samples from first avframe into frame
      i32 samples_to_read = source_frame_size_samples - sample_offset;
      LOG_IF(FATAL, samples_to_read > target_frame_size_samples)
          << "First packet had more samples than target frame size";
      std::memcpy(cur_frame, ((f32*)av_frame->data[0]) + sample_offset,
                  samples_to_read * sizeof(f32));

      samples_so_far += samples_to_read;
      av_frame_unref(av_frame);
      cur_av_frame_idx += 1;

      // Loop through remaining avframes, refilling from packets when necessary
      while (samples_so_far < target_frame_size_samples) {
        if (cur_av_frame_idx == av_frames.size()) {
          av_frames.clear();
          decode_packet(av_frames);
          cur_av_frame_idx = 0;
        }

        av_frame = av_frames[cur_av_frame_idx];
        samples_to_read = std::min(target_frame_size_samples - samples_so_far,
                                   source_frame_size_samples);
        std::memcpy(cur_frame + samples_so_far, (f32*)av_frame->data[0],
                    samples_to_read * sizeof(f32));
        samples_so_far += samples_to_read;
        av_frame_unref(av_frame);
        cur_av_frame_idx += 1;

        VLOG(1) << "Samples to read: " << samples_to_read;
      }

      LOG_IF(FATAL, samples_so_far > target_frame_size_samples)
          << "Got more samples than target";

      // Dealloc all unused avframes
      for (i32 i = cur_av_frame_idx; i < av_frames.size(); ++i) {
        av_frame_unref(av_frames[i]);
      }

      // Save frame
      FrameInfo info(target_frame_size_samples, 1, 1, FrameType::F32);
      frames.push_back(new Frame(info, (u8*) cur_frame));
      cur_frame += target_frame_size_samples;
    }
  }

 private:
  void seek(f64 time) {
    i32 sample_rate = context_->sample_rate;
    i32 source_frame_size_samples = context_->frame_size;

    // Reset codec context
    FF_ERROR(avcodec_send_packet(context_, NULL));
    AVFrame* av_frame = av_frame_alloc();
    while (true) {
      int err = avcodec_receive_frame(context_, av_frame);
      av_frame_unref(av_frame);
      if (err == AVERROR_EOF) {
        break;
      } else {
        FF_ERROR(err);
      }
    }
    avcodec_flush_buffers(context_);

    // wcrichto 7-31-18: in testing, our current ffmpeg setup appears to produce
    // a junk frame from the first packet after a seek when the seeked time
    // isn't 0. So we ask to seek back to one packet before the desired one,
    // then decode the extra packet and drop it.

    // Seek to requested time
    time = std::max(
        time - ((f64)source_frame_size_samples * 2 - 1) / sample_rate, 0.0);
    FF_ERROR(avformat_seek_file(format_context_, stream_->index, INT64_MIN,
                                (i64)(time / time_base_),
                                (i64)(time / time_base_), 0));

    // Flush first packet
    if (std::abs(time) > std::numeric_limits<f64>::epsilon()) {
      VLOG(1) << "Flushing first packet";
      FF_ERROR(av_read_frame(format_context_, &packet_));
      std::vector<AVFrame*> av_frames;
      decode_packet(av_frames);
      for (AVFrame* av_frame : av_frames) {
        av_frame_unref(av_frame);
      }
      av_packet_unref(&packet_);
    }
  }

  void decode_packet(std::vector<AVFrame*>& av_frames) {
    while (av_frames.size() == 0) {
      // Read packets until we get one corresponding to the audio stream
      while (true) {
        FF_ERROR(av_read_frame(format_context_, &packet_));
        if (packet_.stream_index == stream_->index) {
          break;
        }
        av_packet_unref(&packet_);
      }

      // Give packet to decoder asynchronously
      FF_ERROR(avcodec_send_packet(context_, &packet_));

      AVFrame* av_frame = av_frame_alloc();
      LOG_IF(FATAL, av_frame == NULL) << "could not allocate audio frame";

      // Fetch frames from decoder
      while (true) {
        int error = avcodec_receive_frame(context_, av_frame);
        if (error == 0) {
          VLOG(1) << "Frame time: "
                  << av_frame_get_best_effort_timestamp(av_frame) * time_base_
                  << ", offending sample: " << ((f32*)av_frame->data[0])[0];
          LOG_IF(FATAL, av_frame->nb_samples != context_->frame_size)
            << "AVFrame had different # of samples than codec frame size";
          // If decode succeeds, save frame to frame list
          av_frames.push_back(av_frame);
          // Allocate new frame
          av_frame = av_frame_alloc();
          LOG_IF(FATAL, av_frame == NULL) << "could not allocate audio frame";
        } else if (error == AVERROR(EAGAIN)) {
          // We've finished decoding frames from the current packet
          break;
        } else {
          FF_ERROR(error);
        }
      }

      av_frame_unref(av_frame);
      av_packet_unref(&packet_);
    }
  }

  std::string path_;
  std::shared_ptr<StorageBackend> storage_;
  FFStorehouseState file_state_;
  u8* avio_context_buffer_;
  AVFormatContext* format_context_;
  AVIOContext* io_context_;
  AVCodec* codec_;
  AVStream const* stream_;
  AVCodecContext* context_;
  AVPacket packet_;
  f64 time_base_;
  Profiler* profiler_;
};

class AudioEnumerator : public Enumerator {
 public:
  AudioEnumerator(const EnumeratorConfig& config) : Enumerator(config) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse AudioEnumeratorArgs");
      return;
    }

    storage_.reset(StorageBackend::make_from_config(config.storage_config));
    std::string path = args_.path();
    decoder_ = std::make_unique<AudioDecoder>(path, storage_, nullptr);
  }

  i64 total_elements() override {
    double duration = decoder_->duration();
    return (i32)std::floor(duration / args_.frame_size());
  }

  ElementArgs element_args_at(i64 element_idx) override {
    proto::AudioElementArgs args;
    args.set_path(args_.path());
    args.set_frame_size(args_.frame_size());
    size_t size = args.ByteSizeLong();

    ElementArgs element_args;
    element_args.args.resize(size);
    args.SerializeToArray(element_args.args.data(), size);
    element_args.row_id = element_idx;

    return element_args;
  }

 private:
  Result valid_;
  scanner::proto::AudioEnumeratorArgs args_;
  std::unique_ptr<AudioDecoder> decoder_;
  std::shared_ptr<StorageBackend> storage_;
};

class AudioSource : public Source {
 public:
  AudioSource(const SourceConfig& config) : Source(config) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse AudioSourceArgs");
      return;
    }

    storage_.reset(StorageBackend::make_from_config(config.storage_config));
  }

  void read(const std::vector<ElementArgs>& element_args,
            std::vector<Elements>& output_columns) override {
    ProfileBlock _block(profiler_, "audio_read");

    LOG_IF(FATAL, element_args.size() == 0) << "Asked to read zero elements";

    // Deserialize all ElementArgs
    std::vector<i64> row_ids;
    std::string path;
    f64 frame_size;
    for (size_t i = 0; i < element_args.size(); ++i) {
      proto::AudioElementArgs a;
      bool parsed = a.ParseFromArray(element_args[i].args.data(),
                                     element_args[i].args.size());
      LOG_IF(FATAL, !parsed) << "Could not parse element args in AudioSource";
      row_ids.push_back(element_args[i].row_id);
      path = a.path();
      frame_size = a.frame_size();
    }

    // Create a new decoder if the cached one doesn't exist or was for a different video
    if (!decoder_ || last_path_ != path) {
      decoder_.reset(new AudioDecoder(path, storage_, profiler_));
      last_path_ = path;
    }

    // Decode the requested frames
    std::vector<Frame*> frames;
    decoder_->decode(row_ids, frame_size, frames);

    // Insert them into the output stream
    for (Frame* frame : frames) {
      insert_frame(output_columns[0], frame);
    }
  }

 private:
  Result valid_;
  scanner::proto::AudioSourceArgs args_;
  std::string last_path_;
  std::unique_ptr<AudioDecoder> decoder_;
  std::shared_ptr<StorageBackend> storage_;
};

REGISTER_ENUMERATOR(Audio, AudioEnumerator)
    .protobuf_name("AudioEnumeratorArgs");

REGISTER_SOURCE(Audio, AudioSource)
    .output("output")
    .protobuf_name("AudioSourceArgs");

}  // namespace scanner
