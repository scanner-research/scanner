#include "ligthscan/util/caffe.h"

#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <string>

using namespace lightscan;

const int NUM_GPUS = 1;

#define THREAD_RETURN(status__) \
  do {                                           \
    void* val = malloc(sizeof(int));             \
    *((int*)val) = status__;                     \
    return val;                                  \
  } while (0);


void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
}

void master() {
  int num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  printf("%d\n", num_nodes);
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
struct LoadVideoArgs {
  std::string video_path;
};

void* load_video_thread(void* arg) {
  // Setup connection to load video
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously save out results
struct SaveVideoArgs {
};

void* save_video_thread(void* arg) {
  // Setup connection to save video
}

///////////////////////////////////////////////////////////////////////////////
/// Main processing thread that runs the read, evaluate net, write loop
struct ProcessArgs {
  int gpu_device_id;
};

void* process_thread(void* arg) {
  ProcessArgs& args = *reinterpret_cast<ProcessArgs*>(arg);
  int* ret_val = new int;

  // Create IO threads for reading and writing
  pthread_t* load_thread;
  pthread_create(load_thread, NULL, load_video_thread, NULL);

  // pthread_t* save_thread;
  // pthread_create(save_thread, NULL, save_video_thread, NULL);

  // Setup caffe net
  NetInfo net_info = load_neural_net(NetType::VGG, args.gpu_device_id);
  caffe::Net<float>* net = net_info.net;

  // Load
  while (true) {
    // Read batch of frames

    // Decompress batch of frame

    // Process batch of frames

    // Save batch of frames
  }

  // Cleanup
  THREAD_RETURN(EXIT_SUCCESSS);
}

void worker_process() {
  // Parse args to determine video offset

  // Create processing threads for each gpu
  ProcessArgs processing_thread_args[NUM_GPUS];
  pthread_t processing_threads[NUM_GPUS];
  for (int i = 0; i < NUM_GPUS; ++i) {
    processing_thread_args[i].gpu_device_id = i;
    pthread_create(&processing_threads[i],
                   NULL,
                   process_thread,
                   &processing_thread_args[i]);
  }

  // Wait till done
  for (int i = 0; i < NUM_GPUS; ++i) {
    void* result;
    int err = pthread_join(processing_threads[tnum], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join\n");
      pthread_exit(EXIT_FAILURE);
    }

    printf("Joined with thread %d; returned value was %d\n",
           i, *((int *)result));
    free(result);      /* Free memory allocated by thread */
  }

  // Cleanup
}

void shutdown() {
  MPI_Finalize();
}

int main(int argc, char **argv) {
  startup(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    master();
  } else {
    worker_process();
  }

  shutdown();

  return EXIT_SUCCESS;
}

}
