#ifndef DATASET_H
#define DATASET_H
#include"commonHeaders.h"


#define DEBUG_IDXFILE 1

/*  IdxFile
 */
typedef struct _IdxFile
{
    int ndims;
    uint32_t* dims;
    uint8_t* data;
} IdxFile;



typedef struct Dataset{
    IdxFile* images_train;
    IdxFile* labels_train;
    IdxFile* images_test;
    IdxFile* labels_test;
}Data_img;


IdxFile* IdxFile_read(FILE* fp);


/* IdxFile_destroy(self)
   Release the memory.
*/
void IdxFile_destroy(IdxFile* self);


/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile* self, int i);


/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile* self, int i, uint8_t* out);


IdxFile* readDataFile(char* filename);

void readDataset(Data_img* dataset, char* arg1, char* arg2, char* arg3, char* arg4);


void delDataset(Data_img* dataset);


#endif
