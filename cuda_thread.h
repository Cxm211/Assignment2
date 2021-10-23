//
// Created by Cai Xuemeng on 22/10/21.
//

#ifndef A2_STARTER_CODE_CUDA_THREAD_H
#define A2_STARTER_CODE_CUDA_THREAD_H

void GlobalsetValueAt(int *grid, int nRows, int nCols, int row, int col, int val);
int GlobalgetValueAt(const int *grid, int nRows, int nCols, int row, int col);
int goi_cuda(int GRID_X, int GRID_Y, int GRID_Z, int BLOCK_X, int BLOCK_Y, int BLOCK_Z, int nGenerations, const int *startWorld, int nRows, int nCols, int nInvasions, const int *invasionTimes, int **invasionPlans);
#endif //A2_STARTER_CODE_CUDA_THREAD_H