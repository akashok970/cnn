#include"edgeModel.h"

/**
	Initialize the model layers
*/

#define CUT_LAYER 1

// Reduce number of iterations in test and train
#define REDUCE_ITR 0

void initModel(Edge_CNN_arch* e_cnnM){
    /* Initialize layers. */
    /* Input layer - 1x28x28. */
    e_cnnM->linput = Layer_create_input(1, 28, 28);
    /* Conv1 layer - 16x14x14, 3x3 conv, padding=1, stride=2. */
    /* (14-1)*2+3 < 28+1*2 */
    e_cnnM->lconv1 = Layer_create_conv(e_cnnM->linput, 16, 14, 14, 3, 1, 2, 0.1);
#if CUT_LAYER
/* Conv2 layer - 32x7x7, 3x3 conv, padding=1, stride=2. */
    /* (7-1)*2+3 < 14+1*2 */
    e_cnnM->lconv2 = Layer_create_conv(e_cnnM->lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    /* FC1 layer - 200 nodes. */
    e_cnnM->lfull1 = Layer_create_full(e_cnnM->lconv2, 200, 0.1);
    /* FC2 layer - 200 nodes. */
    e_cnnM->lfull2 = Layer_create_full(e_cnnM->lfull1, 200, 0.1);
    /* Output layer - 10 nodes. */
    e_cnnM->loutput = Layer_create_full(e_cnnM->lfull2, 10, 0.1);
#endif
}



/**
	Destroy the model layers
*/
void delModel(Edge_CNN_arch* e_cnnM){

    Layer_destroy(e_cnnM->linput);
    Layer_destroy(e_cnnM->lconv1);
#if CUT_LAYER
	Layer_destroy(e_cnnM->lconv2);
    Layer_destroy(e_cnnM->lfull1);
    Layer_destroy(e_cnnM->lfull2);
    Layer_destroy(e_cnnM->loutput);
#endif
}



void trainM(Edge_CNN_arch* e_cnnM, Data_img* dataset){ 

	IdxFile* images_train = dataset->images_train;
	IdxFile* labels_train = dataset->labels_train;

    fprintf(stderr, "training...\n");
    double rate = 0.1;
    double etotal = 0;
    int batch_size = 32;
	#if 1
		if(images_train == NULL){
			printf("Images train is null \n");
		}
	
	#endif

	int nepoch, train_size;
    #if REDUCE_ITR
    nepoch = 1;
    train_size = 1;
    #else
    nepoch = 10; 
    train_size = images_train->dims[0];
    #endif


	printf("stage 1 \n");
    for (int i = 0; i < nepoch * train_size; i++) {
        /* Pick a random sample from the training data */
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        int index = rand() % train_size;
        
		IdxFile_get3(images_train, index, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        
		int label = IdxFile_get1(labels_train, index);
        
		Layer_setInputs(e_cnnM->linput, x);
        Layer_getOutputs(e_cnnM->loutput, y);
#if 0
        fprintf(stderr, "label=%u, y=[", label);
        for (int j = 0; j < 10; j++) {
            fprintf(stderr, " %.3f", y[j]);
        }
        fprintf(stderr, "]\n");
#endif
        for (int j = 0; j < 10; j++) {
            y[j] = (j == label)? 1 : 0;
        }
        Layer_learnOutputs(e_cnnM->loutput, y);
        etotal += Layer_getErrorTotal(e_cnnM->loutput);
        if ((i % batch_size) == 0) {
            /* Minibatch: update the network for every n samples. */
            Layer_update(e_cnnM->loutput, rate/batch_size);
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d, error=%.4f\n", i, etotal/1000);
            etotal = 0;
        }
    }

	printf("stage 3 \n");

    /* Training finished. */
	
}


#if CUT_LAYER

void testProcess(Edge_CNN_arch* e_cnnM, Data_img* dataset){
	
	IdxFile* images_test = dataset->images_test;
	IdxFile* labels_test = dataset->labels_test;

    fprintf(stderr, "testing...\n");
	int ntests;
    #if REDUCE_ITR
    ntests = 1000;
    #else
    ntests = images_test->dims[0];
    #endif

    int ncorrect = 0;
    for (int i = 0; i < ntests; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        IdxFile_get3(images_test, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        
		Layer_setInputs(e_cnnM->linput, x);
        Layer_getOutputs(e_cnnM->loutput, y);

        int label = IdxFile_get1(labels_test, i);
        
		/* Pick the most probable label. */
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d\n", i);
        }
    }
    fprintf(stderr, "ntests=%d, ncorrect=%d\n", ntests, ncorrect);
}

#endif






