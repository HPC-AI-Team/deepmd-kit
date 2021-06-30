#pragma once

namespace deepmd{

template<typename FPTYPE>
void tabulate_fusion_cpu(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_cpu(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

// transpose
template<typename FPTYPE>
void tabulate_fusion_cpu_transpose(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_cpu_transpose(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_cpu_transpose_sve(
    double * out,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_cpu_transpose_sve(
    float * out,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_grad_cpu_transpose_sve(
    double * dy_dem_x, 
    double * dy_dem,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const double * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_grad_cpu_transpose_sve(
    float * dy_dem_x, 
    float * dy_dem,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const float * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

// packing
template<typename FPTYPE>
void tabulate_fusion_cpu_packing(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_cpu_packing(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_cpu_packing_sve(
    double * out,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_cpu_packing_sve(
    float * out,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_grad_cpu_packing_sve(
    double * dy_dem_x, 
    double * dy_dem,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const double * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_grad_cpu_packing_sve(
    float * dy_dem_x, 
    float * dy_dem,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const float * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);


#if GOOGLE_CUDA
template<typename FPTYPE>
void tabulate_fusion_gpu_cuda(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc,
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_gpu_cuda(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);
#endif // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
template<typename FPTYPE>
void tabulate_fusion_gpu_rocm(
    FPTYPE * out,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const int nloc,
    const int nnei, 
    const int last_layer_size);

template<typename FPTYPE>
void tabulate_fusion_grad_gpu_rocm(
    FPTYPE * dy_dem_x, 
    FPTYPE * dy_dem,
    const FPTYPE * table, 
    const FPTYPE * table_info, 
    const FPTYPE * em_x, 
    const FPTYPE * em, 
    const FPTYPE * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);
#endif // TENSORFLOW_USE_ROCM
}

