#pragma once

namespace deepmd{

// packing
void tabulate_fusion_cpu_packing(
    double * out,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_grad_cpu_packing(
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

void tabulate_fusion_cpu_packing(
    float * out,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size);

void tabulate_fusion_grad_cpu_packing(
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

#ifdef __ARM_FEATURE_SVE

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

#endif
}

