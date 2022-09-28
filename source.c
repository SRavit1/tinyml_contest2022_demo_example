#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include "datatypes.h"
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "3pxnet_fc.h"
#include "3pxnet_cn.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#include "conv1_0_weight.h" 
#include "conv2_0_weight.h" 
#include "31.h" 
#include "fc1_1_running_mean.h" 
#include "fc1_1_running_var.h" 
#include "fc1_1_bias.h" 
#include "fc1_1_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "image.h"
 
#define C1KXY 3
#define C1XY   5
#define C1Z   1
#define C1KZ 32
#define C1PD 0
#define C1PL 1 
static uint8_t l1_act[5] = {1, 2, 3, 4, 5}; //{4, 4, 3, 2, 2};
static int8_t l1wght[96] = {0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0};
static bnDtype bn1thr[32] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
static pckDtype bn1sign[1] = {0xffffffff};

#define C2KXY 3
#define C2XY 3 
#define C2Z 32
#define C2KZ 32
#define C2PD 0
#define C2PL 1 
static pckDtype l2act_bin[3] = {0x58d6ed2, 0x58d6ed2, 0x58d6ed2}; 
static pckDtype l2wght[96] = {0x87c3190d, 0xc9871075, 0x451a81cd, 0x1436e9dd, 0x8c24c146, 0x67cf3543, 0x46fdf762, 0x5e2cba47, 0x22fb9557, 0xec0da7b7, 0x87d4ce94, 0xd7591329, 0x6bc241a2, 0x233d0e2e, 0x79d89495, 0x4b6f3711, 0x3b073246, 0x950ca5bc, 0x6bc3a790, 0x8d3909be, 0xa6d370b7, 0x25886873, 0x589dcef5, 0xeb2f0bba, 0xb9cceb8a, 0xbfda563, 0x5259bd83, 0x3ca94d5d, 0x9f306328, 0xf478b10e, 0xaaa018d2, 0xad4dc25d, 0xc9d14042, 0x8ac896a, 0xa285a6f8, 0xd916870c, 0xe32eb20b, 0x7e722b8f, 0xbf50e941, 0x1579abd4, 0x67a4cd54, 0xfe35bb11, 0x877c91b, 0x1265f308, 0x58ae8862, 0x349be24f, 0xfec671a6, 0x84011fe6, 0xaefe1c29, 0x6328b53f, 0x619a0ed8, 0xe29d67df, 0x58244d8d, 0xf101ee5c, 0x4cb70a43, 0x1b7bc24f, 0xc9baaae3, 0x5d4c17dd, 0x6ce64473, 0x6ca02a75, 0x3469b813, 0x9b46b24d, 0x22d70936, 0xb5c6fc43, 0x8e140df6, 0x89ce0290, 0x2ae1e68a, 0x646a777c, 0xa2433d1d, 0x3a9ea4c1, 0x71a17c58, 0xc587c3e6, 0xc8b96b01, 0x82741762, 0x5f549c8a, 0xc43853b7, 0xfdd8307b, 0x560cd9c5, 0x61613bbd, 0x6a3b738a, 0x68c4f9f0, 0xe91949a7, 0x12804a6e, 0xfa16a9b5, 0x63c4c5c0, 0x299c8271, 0x1ac0f1e4, 0x47a297ef, 0xff5cda5e, 0xfbb1b8ac, 0xda47a27b, 0x4c7f6121, 0xca22b9ff, 0xb06abd49, 0xdaa9473f, 0x3f343900};
static bnDtype bn2thr[32] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} ; 
static pckDtype bn2sign[1] = {0xffffffff} ; 

#define F3I  32
#define F3NPI  0
#define F3O  32
static pckDtype l3act_bin[1] = {0x77a57500};
static pckDtype l3wght[32] = {0x4aedd2f6, 0xf1f73e, 0xa6396308, 0xc40452bf, 0x10dde6ba, 0x38a87575, 0x4a34f6c, 0x6d8d7592, 0x9d58ce47, 0x4040ba8, 0x50485917, 0xc586445f, 0x889e14b4, 0x2aa41f08, 0x7486c01a, 0x68f5724e, 0x8fc908b5, 0xb4e1b16f, 0x2b5e7883, 0x50edd45, 0xc225174c, 0x1dffb9fb, 0xd866d84f, 0x65b12b6c, 0xc4936c76, 0x62a542f9, 0xacede6d2, 0x2c5a2a17, 0x418d1f2e, 0x9af074d5, 0x37a5976, 0x4c0ff52b};
static bnDtype bn3thr[32] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static pckDtype bn3sign[1] = {0xffffffff};

#define F4I 32
#define F4O 10
static pckDtype l4act_bin[1] = {0x27175949};
static pckDtype l4wght[10] = {0x4696c374, 0xf5c475e1, 0xb35d425b, 0x6f78ab3d, 0xeeb34e23, 0x7439186e, 0x83603ca1, 0xc097a7ed, 0x3bde6eab, 0x8caf6bbf};
static bnDtype bn4mean[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static bnDtype bn4var[10] = {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
static bnDtype bn4gamma[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static bnDtype bn4beta[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

static float output[10];

void print_int8Arr(int8_t *arr, int len) {
        for (int i = 0; i < len; i++) {
                printf("%d ", arr[i]);
        }
        printf("\n");
}
void print_uint8Arr(uint8_t *arr, int len) {
        for (int i = 0; i < len; i++) {
                printf("%u ", arr[i]);
        }
        printf("\n");
}
void print_floatArr(float *arr, int len) {
        for (int i = 0; i < len; i++) {
                printf("%f ", arr[i]);
        }
        printf("\n");
}
void print_pckArr(pckDtype *arr, int len) {
        for (int i = 0; i < ceil(len*1.0/pckWdt); i++) {
                for (int j = 0; j < pckWdt && j < len-i*pckWdt; j++) {
                        uint8_t bit = (arr[i]>>(pckWdt-1-j)) & 0x1;
                        //uint8_t bit = (arr[i]>>(j)) & 0x1;
                        printf("%u ", bit);
                }
        }
        printf("\n");
}

int main(){ 
	/*
	printf("x "); print_pckArr(l4act_bin, 32); //print_uint8Arr
	printf("w "); print_pckArr(l4wght, 32); //print_int8Arr
	
	printf("thr "); print_floatArr(bn3thr, 10);
	printf("sign "); print_pckArr(bn3sign, 10);
	
	printf("mean "); print_floatArr(bn4mean, 10);
	printf("var "); print_floatArr(bn4var, 10);
	printf("gamma "); print_floatArr(bn4gamma, 10);
	printf("beta "); print_floatArr(bn4beta, 10);
	*/

	CnBnBwn(l1_act, l1wght, C1Z, C1XY, 1, C1Z, C1KXY, 1, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);
	
	//printf("conv1_out "); print_pckArr(l2act_bin, 32*3);

	CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, 1, C2Z, C2KXY, 1, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign);
	
	//printf("conv2_out "); print_pckArr(l3act_bin, 32);
	
	FcXnorWrap(l3act_bin, l3wght, F3I, F3O, l4act_bin, bn3thr, bn3sign);

	//printf("fc1_out "); print_pckArr(l4act_bin, 32);

	FcXnorNoBinWrap(l4act_bin, l4wght, F4I, F4O, output, bn4mean, bn4var, bn4gamma, bn4beta);

	printf("fc2_out "); print_floatArr(output, 10);
}
