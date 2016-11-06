/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/util/common.h"

namespace scanner {

int PUS_PER_NODE = 1;           // Number of available PUs per node
int GPUS_PER_NODE = 2;          // Number of available GPUs per node
int WORK_ITEM_SIZE = 8;         // Base size of a work item
int TASKS_IN_QUEUE_PER_PU = 4;  // How many tasks per PU to allocate to a node
int LOAD_WORKERS_PER_NODE = 2;  // Number of worker threads loading data
int SAVE_WORKERS_PER_NODE = 2;  // Number of worker threads loading data
int NUM_CUDA_STREAMS = 32;      // Number of cuda streams for image processing
}
