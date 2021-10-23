#include <stdio.h>
#include "exporter.h"

int main()
{
    // Open the file to export to
    FILE *exportFile = fopen("export.out", "w");

    // Let the exporter know about the file it can export to
    initWorldExporter(exportFile);

    // The exporter assumes the world is a 1D array;
    // Edit the function exporterGetValueAt in exporter.cu if you want to use a different layout
    int nRows = 4, nCols = 4;
    int world[] = {
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 4
    };

    // exportWorld will only do stuff after initWorldExporter has been called with a valid file
    exportWorld(world, nRows, nCols);

    // Remember to close the file when you're done
    if (exportFile != NULL)
    {
        fclose(exportFile);
    }
}
