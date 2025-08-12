#include"dataset.h"

uint32_t bswap32(uint32_t x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
}


/* IdxFile_read(fp)
   Reads all the data from given fp.
*/
IdxFile* IdxFile_read(FILE* fp)
{
    /* Read the file header. */
    struct {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
        /* big endian */
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;
#if DEBUG_IDXFILE
    fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
            header.magic, header.type, header.ndims);
#endif
    if (header.magic != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.ndims < 1) return NULL;

    /* Read the dimensions. */
    IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
    if (self == NULL) return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL) return NULL;
    
    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++) {
            
			/* Fix the byte order. */
			uint32_t raw = self->dims[i];
			uint32_t size = bswap32(raw);
            //uint32_t size = be32toh((uint32_t)(self->dims[i]));
            //uint32_t size = self->dims[i];


#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
#endif
            nbytes *= size;
            self->dims[i] = size;
        }
        /* Read the data. */
        self->data = (uint8_t*) malloc(nbytes);
        if (self->data != NULL) {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: read: %u bytes\n", nbytes);
#endif
        }
    }

    return self;
}

/* IdxFile_destroy(self)
   Release the memory.
*/
void IdxFile_destroy(IdxFile* self)
{
    assert (self != NULL);
    if (self->dims != NULL) {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile* self, int i)
{
    assert (self != NULL);
    assert (self->ndims == 1);
    assert (i < self->dims[0]);
    return self->data[i];
}

/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile* self, int i, uint8_t* out)
{
    assert (self != NULL);
    assert (self->ndims == 3);
    assert (i < self->dims[0]);
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i*n], n);
}



IdxFile* readDataFile(char* filename){
    	IdxFile* data_f;
		printf("going to read %s \n", filename);
        FILE* fp = fopen(filename, "rb");
        if (fp == NULL) return NULL;
        data_f = IdxFile_read(fp);
        if (data_f == NULL) return NULL;
        fclose(fp);
	#if 0
		if(data_f != NULL){
			printf("Images train is null in reading file and address is %x\n", &data_f);
		}
	#endif
		return data_f;
}

void readDataset(Data_img* dataset, char* arg1, char* arg2, char* arg3, char* arg4)
{
	dataset->images_train = readDataFile(arg1);
	if(dataset->images_train == NULL){
		printf("data reading failed 1\n");
	}

	dataset->labels_train = readDataFile(arg2);
	if(dataset->labels_train == NULL){
		printf("data reading failed 2\n");
	}

	dataset->images_test = readDataFile(arg3);
	if(dataset->labels_train == NULL){
		printf("data reading failed 3\n");
	}

	dataset->labels_test = readDataFile(arg4);
	if(dataset->labels_train == NULL){
		printf("data reading failed 4\n");
	}
	
}


void delDataset(Data_img* dataset)
{
	IdxFile_destroy(dataset->images_train);
	IdxFile_destroy(dataset->labels_train);
	IdxFile_destroy(dataset->images_test);
	IdxFile_destroy(dataset->labels_test);
}
