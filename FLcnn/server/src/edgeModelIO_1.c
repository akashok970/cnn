#include "edgeModelIO_1.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include "cnn.h"
#define MAGIC "CNN\0"
#define VERSION 1

int is_little_endian() {
    uint16_t x = 1;
    return *((uint8_t*)&x) == 1;
}

void write_int_le(FILE* fp, int val) {
    if (is_little_endian()) {
        fwrite(&val, sizeof(int), 1, fp);
    } else {
        uint8_t bytes[4];
        for (int i = 0; i < 4; ++i)
            bytes[i] = (val >> (i * 8)) & 0xFF;
        fwrite(bytes, 1, 4, fp);
    }
}

int read_int_le(FILE* fp, int* out_val) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, fp) != 4) return 0;
    if (is_little_endian()) {
        memcpy(out_val, bytes, 4);
    } else {
        *out_val = 0;
        for (int i = 0; i < 4; ++i)
            *out_val |= bytes[i] << (i * 8);
    }
    return 1;
}

int deleteFile(const char *filename)
{
    // Check if file exists
    if (access(filename, F_OK) == 0) {
        // File exists, delete it
        if (unlink(filename) != 0) {
            perror("Error deleting existing file");
            return 1;
        }
    }   
    return 0;
}




void xor_encrypt_decrypt(uint8_t* data, size_t len, uint8_t key) {
    for (size_t i = 0; i < len; ++i)
        data[i] ^= key;
}

void save_model_architecture_bin(const char* filename, Layer* head) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { perror("Open arch file"); return; }

    fwrite(MAGIC, 1, 4, fp);
    write_int_le(fp, VERSION);

    int count = 0;
    Layer* temp = head;
    while (temp) { count++; temp = temp->lnext; }
    write_int_le(fp, count);

    temp = head;
    while (temp) {
        write_int_le(fp, temp->ltype);
        write_int_le(fp, temp->depth);
        write_int_le(fp, temp->width);
        write_int_le(fp, temp->height);
        write_int_le(fp, temp->nbiases);
        write_int_le(fp, temp->nweights);
        temp = temp->lnext;
    }

    fclose(fp);
}

void save_model_weights_bin(const char* filename, Layer* head, uint8_t key) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { perror("Open weight file"); return; }

    Layer* temp = head;
    while (temp) {
        size_t bsz = sizeof(double) * temp->nbiases;
        size_t wsz = sizeof(double) * temp->nweights;
        uint8_t* buf_b = malloc(bsz), *buf_w = malloc(wsz);
        memcpy(buf_b, temp->biases, bsz);
        memcpy(buf_w, temp->weights, wsz);
        xor_encrypt_decrypt(buf_b, bsz, key);
        xor_encrypt_decrypt(buf_w, wsz, key);
        fwrite(buf_b, 1, bsz, fp);
        fwrite(buf_w, 1, wsz, fp);
        free(buf_b); free(buf_w);
        temp = temp->lnext;
    }
    fclose(fp);
}

void load_model_weights_bin(const char* filename, Layer* head, uint8_t key) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { perror("Open weight file"); return; }

    Layer* temp = head;
    while (temp) {
        size_t bsz = sizeof(double) * temp->nbiases;
        size_t wsz = sizeof(double) * temp->nweights;
        uint8_t* buf_b = malloc(bsz), *buf_w = malloc(wsz);
        if (fread(buf_b, 1, bsz, fp) != bsz || fread(buf_w, 1, wsz, fp) != wsz) {
            fprintf(stderr, "Failed reading weights\n"); free(buf_b); free(buf_w); fclose(fp); return;
        }
        xor_encrypt_decrypt(buf_b, bsz, key);
        xor_encrypt_decrypt(buf_w, wsz, key);
        memcpy(temp->biases, buf_b, bsz);
        memcpy(temp->weights, buf_w, wsz);
        free(buf_b); free(buf_w);
        temp = temp->lnext;
    }
    fclose(fp);
}



// Send file over socket
int send_file(int sock, const char* fname) {
    FILE* fp = fopen(fname, "rb");
    if(!fp) return -1;
    fseek(fp,0,SEEK_END);
    long sz = ftell(fp);
    fseek(fp,0,SEEK_SET);
    uint32_t net_sz = htonl(sz);
    send(sock, &net_sz, sizeof(net_sz), 0);
    char buf[4096];
    while (1) {
        size_t r = fread(buf,1,sizeof(buf),fp);
        if (r<=0) break;
        send(sock,buf,r,0);
    }
    fclose(fp);
    return 0;
}

// Receive file over socket
int recv_file(int sock, const char* fname) {
    FILE* fp = fopen(fname, "wb");
    if(!fp) return -1;
    uint32_t net_sz;
    if (recv(sock, &net_sz, sizeof(net_sz), MSG_WAITALL) != sizeof(net_sz)) return -1;
    uint32_t sz = ntohl(net_sz);
    char buf[4096];
    uint32_t recvd = 0;
    while (recvd < sz) {
        int toread = sizeof(buf);
        if (sz - recvd < toread) toread = sz - recvd;
        int r = recv(sock, buf, toread, MSG_WAITALL);
        if (r <= 0) break;
        fwrite(buf,1,r,fp);
        recvd += r;
    }
    fclose(fp);
    return (recvd==sz ? 0 : -1);
}

