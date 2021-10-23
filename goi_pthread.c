//
// Created by Cai Xuemeng on 24/9/21.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include "goi_pthread.h"
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
bool isBirthable(int n)
{
    return n == 3;
}

/**
 * Specifies the number(s) of live neighbors of the same faction required for a live cell to remain alive.
 */
bool isSurvivable(int n)
{
    return n == 2 || n == 3;
}

/**
 * Specifies the number of live neighbors of a different faction required for a live cell to die due to fighting.
 */
bool willFight(int n) {
    return n > 0;
}

/**
 * Computes and returns the next state of the cell specified by row and col based on currWorld and invaders. Sets *diedDueToFighting to
 * true if this cell should count towards the death toll due to fighting.
 *
 * invaders can be NULL if there are no invaders.
 */

int getNextState(const int *currWorld, const int *invaders, int nRows, int nCols, int row, int col, bool *diedDueToFighting)
{
    // we'll explicitly set if it was death due to fighting
    *diedDueToFighting = false;

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
    memset(neighborCounts, 0, MAX_FACTIONS * sizeof(int));

    // count neighbors (and self)
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int faction = getValueAt(currWorld, nRows, nCols, row + dy, col + dx);
            if (faction >= DEAD_FACTION)
            {
                neighborCounts[faction]++;
            }
        }
    }

    // we counted this cell as its "neighbor"; adjust for this
    neighborCounts[cellFaction]--;

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

void *thread_task( void* threadData){
    struct thread_data *thread_data_ptr = (struct thread_data*) threadData;
//    printf("%d - ", thread_data_ptr->nRows);
//    printWorld(thread_data_ptr->currWorld,thread_data_ptr->nRows, thread_data_ptr->nCols );
    for (int i = 0; i < thread_data_ptr->numGrid; i++) {
        bool diedDueToFighting;
       // printf("%d - ", thread_data_ptr->nextstate[i]);
        thread_data_ptr->nextstate[i] = getNextState(thread_data_ptr->currWorld, thread_data_ptr->invaders, thread_data_ptr->nRows, thread_data_ptr->nCols, thread_data_ptr->row[i],  thread_data_ptr->col[i],  &diedDueToFighting) ;
        if (diedDueToFighting){
            thread_data_ptr->numDeath += 1;
        }
        //nextState = getNextState(world, inv, nRows, nCols, row, col, &diedDueToFighting);
        //printf("%d - ", thread_data_ptr->nextstate[i]);

    }
}

/**
 * The main simulation logic.
 *
 * goi does not own startWorld, invasionTimes or invasionPlans and should not modify or attempt to free them.
 * nThreads is the number of threads to simulate with. It is ignored by the sequential implementation.
 */
int goi(int nThreads, int nGenerations, const int *startWorld, int nRows, int nCols, int nInvasions, const int *invasionTimes, int **invasionPlans)
{
    //init the pthreads
    pthread_t threads[nThreads];
    int rc;
    struct thread_data threadData[nThreads];

    // death toll due to fighting
    int deathToll = 0;
    int deathTollThreads[nThreads];

    // init the world!
    // we make a copy because we do not own startWorld (and will perform free() on world)
    int *world = malloc(sizeof(int) * nRows * nCols);
    if (world == NULL)
    {
        return -1;
    }

    for (int row = 0; row < nRows; row++)
    {
        for (int col = 0; col < nCols; col++)
        {
            setValueAt(world, nRows, nCols, row, col, getValueAt(startWorld, nRows, nCols, row, col));
        }
    }

#if PRINT_GENERATIONS
    printf("\n=== WORLD 0 ===\n");
    printWorld(world, nRows, nCols);
#endif

#if EXPORT_GENERATIONS
    exportWorld(world, nRows, nCols);
#endif
    for(int t = 0; t < nThreads; t++){
        threadData[t].numDeath = 0;
        threadData[t].numGrid = 0;
    }

    for (int row = 0; row < nRows; row++)
    {
        for (int col = 0; col < nCols; col++)
        {
            int thread_num = ( row * nCols + col  ) % nThreads;
            //  printf("thread : %d \n ", thread_num);
            int num = threadData[thread_num].numGrid;
            //  printf("%d -  ", num);
            threadData[thread_num].row[num] = row;
            threadData[thread_num].col[num] = col;
            threadData[thread_num].nextstate[num] = 0;
            threadData[thread_num].numDeath = 0;
            threadData[thread_num].numGrid+=1;

        }
    }

    // Begin simulating
    int invasionIndex = 0;
    for (int i = 1; i <= nGenerations; i++)
    {
        // is there an invasion this generation?
        int *inv = NULL;
        if (invasionIndex < nInvasions && i == invasionTimes[invasionIndex])
        {
            // we make a copy because we do not own invasionPlans
            inv = malloc(sizeof(int) * nRows * nCols);
            if (inv == NULL)
            {
                free(world);
                return -1;
            }
            for (int row = 0; row < nRows; row++)
            {
                for (int col = 0; col < nCols; col++)
                {
                    setValueAt(inv, nRows, nCols, row, col, getValueAt(invasionPlans[invasionIndex], nRows, nCols, row, col));
                }
            }
            invasionIndex++;
        }

        // create the next world state
        int *wholeNewWorld = malloc(sizeof(int) * nRows * nCols);
        if (wholeNewWorld == NULL)
        {
            if (inv != NULL)
            {
                free(inv);
            }
            free(world);
            return -1;
        }
        for(int t = 0; t < nThreads; t++){
            threadData[t].currWorld = world;
            threadData[t].invaders = inv;
            threadData[t].nCols = nCols;
            threadData[t].nRows = nRows;
            threadData[t].numRead = 0;
            threadData[t].numDeath = 0;
        }

        //create threads
        for(int t = 0; t < nThreads; t++){

            rc = pthread_create(&threads[t], NULL, thread_task, (void *) &threadData[t]);
            if (rc){
                printf("Fail to create thread");
                exit(-1);

            }
        }

        for(int t = 0; t < nThreads; t++){

            pthread_join(threads[t], NULL);
            deathToll+= threadData[t].numDeath;
        }

       //  get new states for each cell
        for (int row = 0; row < nRows; row++)
        {
            for (int col = 0; col < nCols; col++)
            {
                int thread_num = ( row * nCols + col  ) % nThreads;
                int nextState = threadData[thread_num].nextstate[threadData[thread_num].numRead];
                threadData[thread_num].numRead += 1;
                setValueAt(wholeNewWorld, nRows, nCols, row, col, nextState);
            }
        }

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

    free(world);
    return deathToll;
}

