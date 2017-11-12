#include <iostream>
int main() {
  std::cout << "@dirs@" << std::endl;
#ifdef HAVE_CUDA
  std::cout << "-DHAVE_CUDA" << std::endl;
#else
  std::cout << std::endl;
#endif
  return 0;
}
