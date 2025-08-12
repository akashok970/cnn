#ifndef CNN_IO_H
#define CNN_IO_H

#include <stdio.h>
#include <stdint.h>
 #include <unistd.h>
 #include "cnn.h"


int deleteFile(const char *filename);


/* Endianness detection */
int is_little_endian(void);

/* Binary I/O helpers for int */
void write_int_le(FILE* fp, int val);
int read_int_le(FILE* fp, int* out_val);

/* XOR encryption/decryption */
void xor_encrypt_decrypt(uint8_t* data, size_t len, uint8_t key);

/* Save/load model architecture */
void save_model_architecture_bin(const char* filename, Layer* head);
void load_model_architecture_bin(const char* filename, Layer** out_head);

/* Save/load model weights (encrypted) */
void save_model_weights_bin(const char* filename, Layer* head, uint8_t key);
void load_model_weights_bin(const char* filename, Layer* head, uint8_t key);

int recv_file(int sock, const char* fname);
int send_file(int sock, const char* fname);

#endif // CNN_IO_H

