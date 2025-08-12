#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include "edgeModelIO_1.h"
 #include "dataset.h"
 #include "edgeModel.h"



#define PORT 9000
#define SERVER_IP "127.0.0.1"
#define WEIGHT_FILE "../initModel/model_weights.bin"
#define U_WEIGHT_FILE "../wtFiles/umodel_weights.bin"
#define A_WEIGHT_FILE "../wtFiles/amodel_weights.bin"
#define B_WEIGHT_FILE "../wtFiles/bmodel_agg.bin"


#define KEY 0xAA


void train(Edge_CNN_arch* e_cnnM, Data_img* dataset){

    // Train the model
    trainM(e_cnnM, dataset);

    //Save the model
    deleteFile(U_WEIGHT_FILE);
    save_model_weights_bin(U_WEIGHT_FILE, e_cnnM->linput, KEY);
}


void infer(Edge_CNN_arch* e_cnnM, Data_img* dataset){

    load_model_weights_bin(U_WEIGHT_FILE, e_cnnM->linput, KEY);
    // Test the model
    testProcess(e_cnnM, dataset);
}

// Send updated weights to server and receive aggregated weights
int fedLearn(){
    
    // Connect to server
    int sock = socket(AF_INET, SOCK_STREAM, 0); 
    struct sockaddr_in servaddr = {AF_INET, htons(PORT), .sin_addr.s_addr = inet_addr(SERVER_IP)};
    if (connect(sock, (struct sockaddr*)&servaddr, sizeof(servaddr)) != 0) {
        perror("connect");
        return 1;
    }   

    // Send updated weights
    send_file(sock, U_WEIGHT_FILE);

    // Receive aggregated update
    recv_file(sock, A_WEIGHT_FILE);

    close(sock);

	return 0;
}


/* main */
int main(int argc, char* argv[])
{
    if (argc < 4) return 100;

    //Model
    Edge_CNN_arch e_cnnM;

    //Dataset (Train + Test)
    Data_img dataset;

    /* Use a fixed random seed for debugging. */
    srand(0);

    /* Initialize layers. */
    initModel(&e_cnnM);

    // Read dataset
    readDataset(&dataset, argv[1], argv[2], argv[3], argv[4]);

    // Train the model
    //train(&e_cnnM, &dataset);
	
	// After training loaded new weights for testing FL
    load_model_weights_bin(U_WEIGHT_FILE, e_cnnM.linput, KEY);

	fedLearn();

	// Load aggregated model received from server
	// Reload aggregated weights
    load_model_weights_bin(A_WEIGHT_FILE, e_cnnM.linput, KEY);


    // Test the model
    infer(&e_cnnM, &dataset);


    /* Clear the dataset*/
    delDataset(&dataset);

    /* Delete Model. */
    delModel(&e_cnnM);
    return 0;
}

