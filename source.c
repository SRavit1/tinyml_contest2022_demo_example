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
 
#include "weights.h"

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

/*
inArr- packed input array (WC)
outArr- packed output array (WC)
channels- number of input channels (same as num output chanels)
input_len- number of elems in inArr, divisible by channels
kernel_size- size of maxPool kernel
*/
void maxPool1d(pckDtype *inArr, pckDtype *outArr, int channels, int input_len, int kernel_size) {
	int output_len = input_len/kernel_size;
	for (int i = 0; i < output_len; i++) {
		for (int j = 0; j < (channels/pckWdt); j++) {
			pckDtype val = 0;
			for (int k = 0; k < kernel_size; k++) {
				val |= *(inArr + i*(kernel_size*channels/pckWdt) + j + k*(channels/pckWdt));
			}
			*outArr++ = val;
		}
	}
}

int main(){
	printf("x "); print_uint8Arr(l1_act, 32);

	CnBnBwn(l1_act, l1wght, C1Z, C1XY, 1, C1Z, C1KXY, 1, C1KZ, C1PD, C1PL, l2act_bin_prepool, bn1thr, bn1sign);
	printf("conv1_prepool_out "); print_pckArr(l2act_bin_prepool, 32);
	maxPool1d(l2act_bin_prepool, l2act_bin, C1KZ, sizeof(l2act_bin_prepool)/sizeof(pckDtype), 2);
	printf("conv1_out "); print_pckArr(l2act_bin, 32);

	CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, 1, C2Z, C2KXY, 1, C2KZ, l3act_bin_prepool, C2PD, C2PL, bn2thr, bn2sign);
	printf("conv2_prepool_out "); print_pckArr(l3act_bin_prepool, 32);
	maxPool1d(l3act_bin_prepool, l3act_bin, C2KZ, sizeof(l3act_bin_prepool)/sizeof(pckDtype), 2);
	printf("conv2_out "); print_pckArr(l3act_bin, 32);

	CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, 1, C3Z, C3KXY, 1, C3KZ, l4act_bin_prepool, C3PD, C3PL, bn3thr, bn3sign);
	printf("conv3_prepool_out "); print_pckArr(l4act_bin_prepool, 32);
	maxPool1d(l4act_bin_prepool, l4act_bin, C3KZ, sizeof(l4act_bin_prepool)/sizeof(pckDtype), 2);
	printf("conv3_out "); print_pckArr(l4act_bin, 32);

	CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, 1, C4Z, C4KXY, 1, C4KZ, l5act_bin_prepool, C4PD, C4PL, bn4thr, bn4sign);
	printf("conv4_prepool_out "); print_pckArr(l5act_bin_prepool, 32);
	maxPool1d(l5act_bin_prepool, l5act_bin, C4KZ, sizeof(l5act_bin_prepool)/sizeof(pckDtype), 2);
	printf("conv4_out "); print_pckArr(l5act_bin, 32);

	CnXnorWrap(l5act_bin, l5wght, C5Z, C5XY, 1, C5Z, C5KXY, 1, C5KZ, l6act_bin_prepool, C5PD, C5PL, bn5thr, bn5sign);
	printf("conv5_prepool_out "); print_pckArr(l6act_bin_prepool, 32);
	maxPool1d(l6act_bin_prepool, l6act_bin, C5KZ, sizeof(l6act_bin_prepool)/sizeof(pckDtype), 2);
	printf("conv5_out "); print_pckArr(l6act_bin, 32);
	
	FcXnorWrap(l6act_bin, l6wght, F6I, F6O, l7act_bin, bn6thr, bn6sign);
	printf("fc1_out "); print_pckArr(l7act_bin, 32);

	FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn7mean, bn7var, bn7gamma, bn7beta);
	printf("fc2_out "); print_floatArr(output, F7O);
}