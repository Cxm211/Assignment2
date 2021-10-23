#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include "cuda_thread.h"
#include "util.h"
#include "exporter.h"
#include "settings.h"

// including the "dead faction": 0
#define MAX_FACTIONS 10

// this macro is here to make the code slightly more readable, not because it can be safely changed to
// any integer value; changing this to a non-zero value may break the code
#define DEAD_FACTION 0

/**
 * Specifies the number(s) of live neighbors of the same faction required for a dead cell to become alive.
 */
__device__  int death[1000];

__device__ bool isBirthable(int n)
{
    return n == 3;
}

/**
 * Specifies the number(s) of live neighbors of the same faction required for a live cell to remain alive.
 */
__device__ bool isSurvivable(int n)
{
    return n == 2 || n == 3;
}

/**
 * Specifies the number of live neighbors of a different faction required for a live cell to die due to fighting.
 */
__device__ bool willFight(int n) {
    return n > 0;
}



void GlobalsetValueAt(int *grid, int nRows, int nCols, int row, int col, int val)
{
    if (row < 0 || row >= nRows || col < 0 || col >= nCols)
    {
        return;
    }

    *(grid + (row * nCols) + col) = val;
}


int GlobalgetValueAt(const int *grid, int nRows, int nCols, int row, int col)
{
    if (row < 0 || row >= nRows || col < 0 || col >= nCols)
    {
        return -1;
    }

    return *(grid + (row * nCols) + col);
}
/**
 * Computes and returns the next state of the cell specified by row and col based on currWorld and invaders. Sets *diedDueToFighting to
 * true if this cell should count towards the death toll due to fighting.
 *
 * invaders can be NULL if there are no invaders.
 */

__device__ int getNextState(const int *currWorld, const int *invaders, int nRows, int nCols, int row, int col, bool *diedDueToFighting)
{
    // we'll explicitly set if it was death due to fighting
    *diedDueToFighting = false;
    cudaError_t rc;
    // faction of this cell
    int cellFaction = getValueAt(currWorld, nRows, nCols, row, col);

    // did someone just get landed on?
    if (invaders != NULL && getValueAt(invaders, nRows, nCols, row, col) != DEAD_FACTION)
    {
        *diedDueToFighting = cellFaction != DEAD_FACTION;
        return getValueAt(invaders, nRows, nCols, row, col);
    }

    // tracks count of each faction adjacent to this cell
    int neighborCounts[MAX_FACTIONS];
    for(int i = 0; i < MAX_FACTIONS; i++){
        neighborCounts[i] = 0;
    }
   // memset(neighborCounts, 0, MAX_FACTIONS * sizeof(int));

    // count neighbors (and self)
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int faction = getValueAt(currWorld, nRows, nCols, row + dy, col + dx);
            if (faction >= DEAD_FACTION)
            {
                neighborCounts[faction]++;
               //printf("NEIGHBOR: %d", neighborCounts[faction]);
            }
        }
    }

    // we counted this cell as its "neighbor"; adjust for this
    neighborCounts[cellFaction]--;
//    for(int i = 0; i < MAX_FACTIONS; i++){
//        printf("N: %d", neighborCounts[i]);
//    }
    if (cellFaction == DEAD_FACTION)
    {
        // this is a dead cell; we need to see if a birth is possible:
        // need exactly 3 of a single faction; we don't care about other factions

        // by default, no birth
        int newFaction = DEAD_FACTION;

        // start at 1 because we ignore dead neighbors
        for (int faction = DEAD_FACTION + 1; faction < MAX_FACTIONS; faction++)
        {
            int count = neighborCounts[faction];
//            printf("COUNT: %d", count);
            if (isBirthable(count))
            {
                newFaction = faction;
            }
        }

        return newFaction;
    }
    else
    {
        /**
         * this is a live cell; we follow the usual rules:
         * Death (fighting): > 0 hostile neighbor
         * Death (underpopulation): < 2 friendly neighbors and 0 hostile neighbors
         * Death (overpopulation): > 3 friendly neighbors and 0 hostile neighbors
         * Survival: 2 or 3 friendly neighbors and 0 hostile neighbors
         */

        int hostileCount = 0;
        for (int faction = DEAD_FACTION + 1; faction < MAX_FACTIONS; faction++)
        {
            if (faction == cellFaction)
            {
                continue;
            }
            hostileCount += neighborCounts[faction];
        }

        if (willFight(hostileCount))
        {
            *diedDueToFighting = true;
            return DEAD_FACTION;
        }

        int friendlyCount = neighborCounts[cellFaction];
        if (!isSurvivable(friendlyCount))
        {
            return DEAD_FACTION;
        }

        return cellFaction;
    }
}



__global__ void execute( int * wholeNewWorld, const int *currWorld, const int *invaders, int nRows, int nCols ){
    int tid = (threadIdx.z * blockDim.y * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y ) + (blockDim.x * blockDim.y * blockDim.z) * ( blockIdx.x * gridDim.y + blockIdx.y +  blockIdx.z * gridDim.x * gridDim.y );
    int num = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    for (int row = 0; row < nRows; row++)
    {
        for (int col = 0; col < nCols; col++)
        {
            int id = row * nCols + col;
            bool diedDueToFighting;
            if ( id % num == tid){
                //printf("tid: %d , id: %d\n", tid, id);
                int nextState = getNextState(currWorld, invaders, nRows, nCols, row, col, &diedDueToFighting);
               // printf("NEXT: %d\n", nextState);
                setValueAt(wholeNewWorld, nRows, nCols, row, col, nextState);
                if (diedDueToFighting){
                    death[tid]++;
                }
                diedDueToFighting = false;
            }
        }
    }
}

/**
 * The main simulation logic.
 *
 * goi does not own startWorld, invasionTimes or invasionPlans and should not modify or attempt to free them.
 * nThreads is the number of threads to simulate with. It is ignored by the sequential implementation.
 */
int goi_cuda(int GRID_X, int GRID_Y, int GRID_Z, int BLOCK_X, int BLOCK_Y, int BLOCK_Z, int nGenerations, const int *startWorld, int nRows, int nCols, int nInvasions, const int *invasionTimes, int **invasionPlans)
{
    // death toll due to fighting
    int deathToll = 0;
    int num = GRID_X * GRID_Y * GRID_Z * BLOCK_X * BLOCK_Y * BLOCK_Z;

//    int death[num];
//    int* deathNum;

    // init the world!
    // we make a copy because we do not own startWorld (and will perform free() on world)
    int *world = static_cast<int *>(malloc(sizeof(int) * nRows * nCols));
    if (world == NULL)
    {
        return -1;
    }
    for (int row = 0; row < nRows; row++)
    {
        for (int col = 0; col < nCols; col++)
        {
            GlobalsetValueAt(world, nRows, nCols, row, col, GlobalgetValueAt(startWorld, nRows, nCols, row, col));
        }
    }

#if PRINT_GENERATIONS
    printf("\n=== WORLD 0 ===\n");
    printWorld(world, nRows, nCols);
#endif

#if EXPORT_GENERATIONS
    exportWorld(world, nRows, nCols);
#endif

    // Begin simulating
    int invasionIndex = 0;
    for (int i = 1; i <= nGenerations; i++)
    {
        // is there an invasion this generation?
        int *inv = NULL;
        if (invasionIndex < nInvasions && i == invasionTimes[invasionIndex])
        {
            // we make a copy because we do not own invasionPlans
            inv = static_cast<int *>(malloc(sizeof(int) * nRows * nCols));
            if (inv == NULL)
            {
                free(world);
                return -1;
            }
            for (int row = 0; row < nRows; row++)
            {
                for (int col = 0; col < nCols; col++)
                {
                    GlobalsetValueAt(inv, nRows, nCols, row, col, GlobalgetValueAt(invasionPlans[invasionIndex], nRows, nCols, row, col));
                }
            }
            invasionIndex++;
        }

        // create the next world state
        int *wholeNewWorld = static_cast<int *>(malloc(sizeof(int) * nRows * nCols));
        if (wholeNewWorld == NULL)
        {
            if (inv != NULL)
            {
                free(inv);
            }
            free(world);
            return -1;
        }
        for (int row = 0; row < nRows; row++)
        {
            for (int col = 0; col < nCols; col++)
            {
                GlobalsetValueAt(wholeNewWorld, nRows, nCols, row, col, 0);
            }
        }

//        cudaMalloc((void**)&deathNum, num);
//        cudaMemcpy(deathNum, death, num, cudaMemcpyHostToDevice);
//
//        printf("HAHA\n");
//        printWorld(world,  nRows,  nCols);
//        printWorld(wholeNewWorld,  nRows,  nCols);
        int *wholeNewWorldCuda;
        int *worldCuda;
        int *invCuda;

        cudaMalloc((void**)&wholeNewWorldCuda, sizeof(int) * nRows * nCols);
        cudaMalloc((void**)&worldCuda, sizeof(int) * nRows * nCols);
        cudaMalloc((void**)&invCuda, sizeof(int) * nRows * nCols);

        cudaMemcpy(wholeNewWorldCuda, wholeNewWorld, sizeof(int) * nRows * nCols, cudaMemcpyHostToDevice);
        cudaMemcpy(worldCuda, world, sizeof(int) * nRows * nCols, cudaMemcpyHostToDevice);
        cudaMemcpy(invCuda, inv, sizeof(int) * nRows * nCols, cudaMemcpyHostToDevice);

        dim3 gridDim(GRID_X,GRID_Y,GRID_Z);
        dim3 blockDim(BLOCK_X,BLOCK_Y, BLOCK_Z);

        execute<<<gridDim, blockDim>>>(wholeNewWorldCuda, worldCuda, invCuda, nRows, nCols);
        cudaDeviceSynchronize();

        cudaMemcpy(wholeNewWorld, wholeNewWorldCuda, sizeof(int) * nRows * nCols, cudaMemcpyDeviceToHost);
        cudaMemcpy(world, worldCuda, sizeof(int) * nRows * nCols, cudaMemcpyDeviceToHost);
        cudaMemcpy(inv, invCuda, sizeof(int) * nRows * nCols, cudaMemcpyDeviceToHost);

//        cudaMemcpy(death, deathNum, num, cudaMemcpyDeviceToHost);
        // get new states for each cell
//        for (int row = 0; row < nRows; row++)
//        {
//            for (int col = 0; col < nCols; col++)
//            {
//                bool diedDueToFighting;
//                int nextState = getNextState(world, inv, nRows, nCols, row, col, &diedDueToFighting);
//                setValueAt(wholeNewWorld, nRows, nCols, row, col, nextState);
//                if (diedDueToFighting)
//                {
//                    deathToll++;
//                }
//            }
//        }

        if (inv != NULL)
        {
            free(inv);
        }

        // swap worlds
        free(world);
        world = wholeNewWorld;

#if PRINT_GENERATIONS
        printf("\n=== WORLD %d ===\n", i);
        printWorld(world, nRows, nCols);
#endif

#if EXPORT_GENERATIONS
        exportWorld(world, nRows, nCols);
#endif
    }
    int host_death[1000];
    cudaError_t rc = cudaMemcpyFromSymbol(&host_death, death, sizeof(death));

    if (rc != cudaSuccess)
    {
        printf("Could not copy from device. Reason: %s\n", cudaGetErrorString(rc));
    }
    for (int i = 0; i < num; i++){
        deathToll += host_death[i];
    }
    free(world);
    return deathToll;
}
