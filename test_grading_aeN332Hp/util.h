#ifndef UTIL_H
#define UTIL_H

__device__ int getValueAt(const int *grid, int nRows, int nCols, int row, int col);
__device__ void setValueAt(int *grid, int nRows, int nCols, int row, int col, const int val);
void printWorld(const int *world, int nRows, int nCols);

#endif
