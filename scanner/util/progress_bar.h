#pragma once

#include "scanner/util/common.h"
#include <sys/ioctl.h>

#include <cstring>
#include <iomanip>
#include <iostream>

#define TOTAL_PERCENTAGE 100.0
#define CHARACTER_WIDTH_PERCENTAGE 4

namespace scanner {

class ProgressBar {
 public:
  ProgressBar();
  ProgressBar(u64 n_, const char* description_ = "",
              std::ostream& out_ = std::cerr);

  void SetFrequencyUpdate(u64 frequency_update_);
  void SetStyle(const char* unit_bar_, const char* unit_space_);

  void Progressed(u64 idx_);

 private:
  u64 n;
  unsigned int desc_width;
  u64 frequency_update;
  std::ostream* out;

  const char* description;
  const char* unit_bar;
  const char* unit_space;

  void ClearBarField();
  int GetConsoleWidth();
  int GetBarLength();
};

}
