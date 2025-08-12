#ifndef EDGE_MODEL_H
#define EDGE_MODEL_H

#include "cnn.h"
#include "dataset.h"


typedef struct Edge_CNN_architecture{
    /* Input layer  */
    Layer* linput;
    // Conv1 layer
    Layer* lconv1;
	// Conv2 layer
    Layer* lconv2;
    // FC1 layer
    Layer* lfull1;
    // FC2 layer
    Layer* lfull2;
    // Output layer
    Layer* loutput;

}Edge_CNN_arch;

/**
    Initialize the model layers
*/
void initModel(Edge_CNN_arch* e_cnnM);

/**
    Destroy the model layers
*/
void delModel(Edge_CNN_arch* e_cnnM);

/**
    Train the model
*/
void trainM(Edge_CNN_arch* e_cnnM, Data_img* dataset);


/**
    Test the model
*/
void testProcess(Edge_CNN_arch* e_cnnM, Data_img* dataset);

#endif
