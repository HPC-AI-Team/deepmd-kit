#include "fast_tanh.h"
#include "device.h"

template <typename FPTYPE>
__global__ void fast_tanh(
    const FPTYPE* inputs,FPTYPE* outputs,size_t N) 
{
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  outputs[idx] = tanh(inputs[idx]);
}



namespace deepmd{


template<typename FPTYPE>
void fast_tanh_cuda(const FPTYPE* inputs,FPTYPE* outputs,size_t N){
    const int THREAD_ITEMS = 1024;
    const int BLOCK_NUMS = (N + THREAD_ITEMS - 1) / THREAD_ITEMS;
    fast_tanh<<<BLOCK_NUMS, THREAD_ITEMS>>>(inputs, outputs, N);
}

template void fast_tanh_cuda<float>(const float* inputs,float* outputs,size_t N);
template void fast_tanh_cuda<double>(const double* inputs,double* outputs,size_t N);


}
