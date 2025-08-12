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

#define CLIENT_COUNT 6
#define KEY 0xAA

int client_sockets[CLIENT_COUNT];

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

// let all clients be connected (timeout nmeed to be added later)
void receiveConnectionfrmClients(int* sockfd)
{
	*sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in servaddr = {AF_INET, htons(PORT), .sin_addr.s_addr = INADDR_ANY};
    bind(*sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));
    listen(*sockfd, CLIENT_COUNT);
    printf("Waiting for %d clients...\n", CLIENT_COUNT);

    // Accept all clients first
    for (int i = 0; i < CLIENT_COUNT; ++i) {
        client_sockets[i] = accept(*sockfd, NULL, NULL);
        printf("Client %d connected\n", i + 1);
    }

}

// receive weights from connected clients
void receiveWt()
{
    // Receive all weight files
    for (int i = 0; i < CLIENT_COUNT; ++i) {
        char fname[64];
        snprintf(fname, sizeof(fname), "./files/client_%d_weights.bin", i);
        deleteFile(fname);
        if (recv_file(client_sockets[i], fname) != 0) {
            fprintf(stderr, "Error receiving file from client %d\n", i + 1);
        } else {
            printf("Received weights from client %d\n", i + 1);
        }
    }
}

// Aggregates weights from multiple files into `global`
void aggregate_all(Layer* global) {
    
	//temporary Model to load client models
    Edge_CNN_arch tmp_cnnM;
    
	/* Initialize layers. */
    initModel(&tmp_cnnM);
	
	
	Layer* tmp = tmp_cnnM.linput;
    for (int i = 0; i < CLIENT_COUNT; ++i) {
        char fname[64];
        snprintf(fname, sizeof(fname), "./files/client_%d_weights.bin", i);
        //load_model_architecture_bin(ARCH_FILE, &tmp);
        load_model_weights_bin(fname, tmp, KEY);

        Layer *g = global, *t = tmp;
        while (g && t) {
            for (int j = 0; j < g->nweights; ++j)
                g->weights[j] += t->weights[j];
            for (int j = 0; j < g->nbiases; ++j)
                g->biases[j] += t->biases[j];
            g = g->lnext;
            t = t->lnext;
        }
    }

    // Divide by number of clients to average
    Layer *g = global;
    while (g) {
        for (int i = 0; i < g->nweights; ++i)
            g->weights[i] /= CLIENT_COUNT;
        for (int i = 0; i < g->nbiases; ++i)
            g->biases[i] /= CLIENT_COUNT;
        g = g->lnext;
    }

    deleteFile(A_WEIGHT_FILE);
    // Save aggregated model
    save_model_weights_bin(A_WEIGHT_FILE, global, KEY);

    /* Delete Model. */
    delModel(&tmp_cnnM);
	
	printf("Aggregation complete\n");
}

//Send weights back to clients after aggregation
void sndWt2Clients(){
	
	 // Send back aggregated model to each client
    for (int i = 0; i < CLIENT_COUNT; ++i) {
        send_file(client_sockets[i], A_WEIGHT_FILE);
        printf("Sent aggregated weights to client %d\n", i + 1);
        close(client_sockets[i]);
    }


}


// collect weights from clients and aggregate then using FedAvg. Send back aggregated weights for the model update to all clients.
int fedLearnServer(Layer* global){
    
	//server socket
	int sockfd=0;

	// let clients connect to server for sharing weight updates
	receiveConnectionfrmClients(&sockfd);

	// receive weights from connected clients
	receiveWt();

	// perform aggregation
	aggregate_all(global);

	//Send weights back to clients after aggregation
	sndWt2Clients();
    
	//close server socket
	close(sockfd);
	
	return 0;
}


/* main */
int main(int argc, char* argv[])
{

	if(argc<4){
		return 400;
	}
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

	// fed ml from server side	
	fedLearnServer(e_cnnM.linput);


    /* Clear the dataset*/
    delDataset(&dataset);

    /* Delete Model. */
    delModel(&e_cnnM);
    return 0;
}

