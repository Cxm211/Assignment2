//
// Created by Cai Xuemeng on 24/9/21.
//

#ifndef A1_CODE_GOI_PTHREAD_H
#define A1_CODE_GOI_PTHREAD_H

int goi_pthread(int nThreads, int nGenerations, const int *startWorld, int nRows, int nCols, int nInvasions, const int *invasionTimes, int **invasionPlans);

struct thread_data{
    const int *currWorld;
    const int *invaders;
    int nRows;
    int nCols;
    int row[1000];
    int col[1000];
    int numGrid;
    int numRead;
    int numDeath;
    int nextstate[1000];
};

#endif //A1_CODE_GOI_PTHREAD_H
