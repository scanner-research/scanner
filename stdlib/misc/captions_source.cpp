#include "scanner/api/enumerator.h"
#include "scanner/api/source.h"
#include "scanner/util/json.hpp"
#include "stdlib/stdlib.pb.h"

namespace scanner {

using nlohmann::json;
using storehouse::StoreResult;

typedef struct {
  i32 index;
  f32 start;
  f32 end;
  std::string line;
} Caption;

// https://github.com/nlohmann/json#basic-usage
void to_json(json& j, const Caption& c) {
  j = json{
      {"index", c.index}, {"start", c.start}, {"end", c.end}, {"line", c.line}};
}

class CaptionsParser {
 public:
  CaptionsParser(const std::string path,
                 std::shared_ptr<storehouse::StorageBackend> storage)
    : path(path) {
    std::unique_ptr<storehouse::RandomReadFile> file;
    StoreResult result;
    EXP_BACKOFF(
        storehouse::make_unique_random_read_file(storage.get(), path, file),
        result);
    LOG_IF(FATAL, result != StoreResult::Success)
        << "Could not open captions file " << path;

    size_t size;
    result = file->get_size(size);
    LOG_IF(FATAL, result != StoreResult::Success)
        << "Could not get size of " << path;

    u8* buffer = new u8[size];
    u64 pos = 0;
    s_read(file.get(), buffer, size, pos);
    const std::string buffer_s((char*)buffer, size);

    parse_captions(buffer_s);
  }

  std::vector<Caption> find(f32 start, f32 end) {
    std::vector<Caption> output;
    for (auto& cap : captions_) {
      if (cap.start >= start && cap.start < end) {
        output.push_back(cap);
      }
    }
    return output;
  }

  std::string path;

 private:
  // Adapted from
  // https://github.com/saurabhshri/simple-yet-powerful-srt-subtitle-parser-cpp
  std::vector<std::string>& split(const std::string& s, char delim,
                                  std::vector<std::string>& elems) {
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
      elems.push_back(item);
    }
    return elems;
  }

  f32 parse_timestamp(const std::string& timestamp) {
    std::vector<std::string> parts;
    split(timestamp, ':', parts);

    std::vector<std::string> subparts;
    split(parts[2], ',', subparts);

    return std::stoi(parts[0]) * 3600. + std::stoi(parts[1]) * 60. +
           std::stoi(subparts[0]) * 1. + (std::stoi(subparts[1]) / 1000.);
  }

  void parse_captions(const std::string& buffer) {
    std::istringstream infile(buffer);
    std::string line, completeLine = "", timeLine = "";
    f32 start;
    f32 end;
    int subNo, turn = 0;

    /*
     * turn = 0 -> Add subtitle number
     * turn = 1 -> Add string to timeLine
     * turn > 1 -> Add string to completeLine
     */

    while (std::getline(infile, line)) {
      line.erase(remove(line.begin(), line.end(), '\r'), line.end());

      if (line.compare("")) {
        if (!turn) {
          subNo = atoi(line.c_str());
          turn++;
          continue;
        }

        if (line.find("-->") != std::string::npos) {
          timeLine += line;

          std::vector<std::string> srtTime;
          srtTime = split(timeLine, ' ', srtTime);
          start = parse_timestamp(srtTime[0]);
          end = parse_timestamp(srtTime[2]);
        } else {
          if (completeLine != "") completeLine += " ";

          completeLine += line;
        }

        turn++;
      }

      else {
        turn = 0;
        captions_.push_back(Caption{subNo, start, end, completeLine});
        completeLine = timeLine = "";
      }

      // insert last remaining subtitle
      if (infile.eof()) {
        captions_.push_back(Caption{subNo, start, end, completeLine});
      }
    }
  }

  std::unique_ptr<storehouse::RandomReadFile> file_;
  std::vector<Caption> captions_;
};

class CaptionsEnumerator : public Enumerator {
 public:
  CaptionsEnumerator(const EnumeratorConfig& config) : Enumerator(config) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse CaptionsEnumeratorArgs");
      return;
    }
  }

  i64 total_elements() override {
    return std::floor((args_.max_time() - 1) / args_.window_size());
  }

  ElementArgs element_args_at(i64 element_idx) override {
    proto::CaptionsElementArgs args;
    args.set_path(args_.path());
    args.set_window_size(args_.window_size());

    size_t size = args.ByteSizeLong();
    ElementArgs element_args;
    element_args.args.resize(size);
    args.SerializeToArray(element_args.args.data(), size);
    element_args.row_id = element_idx;

    return element_args;
  }

 private:
  Result valid_;
  scanner::proto::CaptionsEnumeratorArgs args_;
};

class CaptionsSource : public Source {
 public:
  CaptionsSource(const SourceConfig& config)
    : Source(config), parser_(nullptr) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse CaptionsSourceArgs");
      return;
    }

    storage_.reset(
        storehouse::StorageBackend::make_from_config(config.storage_config));
  }

  void read(const std::vector<ElementArgs>& element_args,
            std::vector<Elements>& output_columns) override {
    LOG_IF(FATAL, element_args.size() == 0) << "Asked to read zero elements";

    // Deserialize all ElementArgs
    std::string path;
    double window_size;
    std::vector<i64> row_ids;
    for (size_t i = 0; i < element_args.size(); ++i) {
      proto::CaptionsElementArgs a;
      bool parsed = a.ParseFromArray(element_args[i].args.data(),
                                     element_args[i].args.size());
      LOG_IF(FATAL, !parsed) << "Could not parse element args in Captions";

      row_ids.push_back(element_args[i].row_id);
      path = a.path();
      window_size = a.window_size();
    }

    // If the file changed, create a new parser
    if (!parser_ || parser_->path != path) {
      parser_ = std::make_unique<CaptionsParser>(path, storage_);
    }

    // Find and serialize all subtitles in given window
    std::vector<std::string> bufs;
    std::vector<size_t> sizes;
    size_t total_size = 0;
    for (i64 row_id : row_ids) {
      double start = row_id * window_size;
      double end = (row_id + 1) * window_size;
      std::vector<Caption> captions = parser_->find(start, end);
      json j = captions;
      std::string jstr = j.dump();
      total_size += jstr.size();
      sizes.push_back(jstr.size());
      bufs.push_back(jstr);
    }

    u8* block_buffer = new_block_buffer(CPU_DEVICE, total_size, sizes.size());
    u8* cursor = block_buffer;
    for (i32 i = 0; i < sizes.size(); ++i) {
      memcpy_buffer(cursor, CPU_DEVICE, (u8*)bufs[i].c_str(), CPU_DEVICE,
                    sizes[i]);
      insert_element(output_columns[0], cursor, sizes[i]);
      cursor += sizes[i];
    }
  }

 private:
  Result valid_;
  scanner::proto::CaptionsSourceArgs args_;
  std::shared_ptr<storehouse::StorageBackend> storage_;
  std::unique_ptr<CaptionsParser> parser_;
};

REGISTER_ENUMERATOR(Captions, CaptionsEnumerator)
    .protobuf_name("CaptionsEnumeratorArgs");

REGISTER_SOURCE(Captions, CaptionsSource)
    .output("output")
    .protobuf_name("CaptionsSourceArgs");

}  // namespace scanner
