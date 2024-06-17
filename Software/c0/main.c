#include "main.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "image.h"
#include "io.h"
#include "nios2.h"
#include "stddef.h"
#include "stdint.h"
#include "unistd.h"
#include "string.h"
#include "system.h"

// Register access for custom hardware clock counter
#define CCOUNTER_BASE (CLOCK_COUNTER_0_BASE)
#define CCOUNTER_RESET() IOWR(CCOUNTER_BASE, 0, 0)
#define CCOUNTER_GETL() IORD(CCOUNTER_BASE, 0)
#define CCOUNTER_GETH() IORD(CCOUNTER_BASE, 4)
#define CCOUNTER_CAPTURE() (CCOUNTER_GETL() | ((uint64_t)(CCOUNTER_GETH()) << 32))

unsigned char *ch_images;  // Images data region
float *fp_weights; // Network weights data region
float *fp_image; // Scaled floating-point image to be processed

float *matA;  // Auxiliary matrix A
float *matAT;  // Transpose of matA
float *matB;   // Auxiliary matrix B of 22 feature maps with 5*5 weights each
float *matBT;  // Transpose of matB
float *matC;   // Auxiliary matrix C with intermediate output (before adding bias) in convolutional layer
float *matCT;   // Transpose of matC
float *matCbias; // Output of convolutional layer 22 images of size 24*24
float *matCpool; // Output of pooling layer 22 images of size 12*12
float *matConn;  // Intermediate output (before adding bias) of fully connected layer (10 elements)
float *matConnB; // Output of fully connected layer (10 elements)
float *matSoftM; // Output of softmax layer (10 elements)

// Prints first <size> elements of matrix f
void print_fp(float *f, int size, char *c) {
  int i;
  printf("%s\n", c);
  for (i = 0; i < size; i++) {
    if ((i % 12) == 0) printf("%02d: ", i/12);
    printf("%f ", f[i]);
    if ((i % 12) == 11) printf("\n");
  }
  printf("\n");
}

// Prints matrix elements
void print_fp_mat(float *mat, int rows, int cols) {
  int i, j;

  for (i=0; i<rows; i++) {
    for (j=0; j<cols; j++) {
      printf("%f ", mat[i*cols+j]);
    }
    printf("\n");
  }

}

// Matrix multiplication: C = A * B
void gemm(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
  int i, j, k;

  for (i=0; i<rowsA; i++) {
    for (j=0; j<colsB; j++) {
      C[i*colsB+j] = 0.0;
      for (k=0; k<colsA; k++) {
	      C[i*colsB+j] += A[i*colsA+k] * B[k*colsB+j];
      }
    }
  }
}

// Matrix multiplication: C = A * transposed(B)
void gemmBT(float *A, float *B, float *C, int rowsA, int colsA, int rowsB) {
  int i, j, k;
  int colsBT, colsB;

  colsBT = rowsB;
  colsB = colsA;
  for (i=0; i<rowsA; i++) {
    for (j=0; j<colsBT; j++) {
      C[i*colsBT+j] = 0.0;
      for (k=0; k<colsA; k++) {
	      C[i*colsBT+j] += A[i*colsA+k] * B[j*colsB+k];
      }
    }
  }
}

// Transposes matrix
void transpose(float *C, int rows, int cols, float *CT) {
  int i, j;

  for (i=0; i<rows; i++) {
    for (j=0; j<cols; j++) {
      CT[j*rows+i] = C[i*cols+j] ;
    }
  }
  
}

// Adds bias to matrix C
// if (transpose_flag == 1) also transposes input matrix
void add_bias(float *C, int rows, int cols, float *bias, float *Cbias, int transpose_flag) {
  int i, j;

  if (transpose_flag) {
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        Cbias[j*rows+i] = C[i*cols+j] + bias[j] ;
      }
    }
  }
  else {
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
	    Cbias[i*cols+j] = C[i*cols+j] + bias[i] ;
      }
    }
  }
}

// Prepares matrix A to calculate convolutions as a matrix multiplication
void prepare_matrixA() {
  int i, j, k, row, col;

  for (i=0; i<24; i++) {
    for (j=0; j<24; j++) {
      for (k=0; k<25; k++) {
        row = i + k/5;
        col = j + (k%5);
        // Matrix A has (24*24) rows and 25 columns
        matA[(i*24+j)*25+k] = fp_image[row*IMAGE_WIDTH+col];
      }
    }
  }
}

// Returns the position of the largest value in the 10-element input vector
// The softmax function normalizes the input-vector to a probability distribution
int forward_softmax_layer() {
  int i, n=10, best=-1;
  float sum = 0.0, e;
  float largest = -FLT_MAX;

  for(i = 0; i < n; ++i){
    if(matConnB[i] > largest) {
      largest = matConnB[i];
      best = i;
    }
  }
	
  for(i = 0; i < n; ++i){
    e = exp(matConnB[i] - largest);
    sum += e;
    matSoftM[i] = e;
  }
  for(i = 0; i < n; ++i){
    matSoftM[i] /= sum;
  }
  // print_fp((float *)matSoftM, 10, "Softmax");

  return best;
}

// The max-pool layer condenses the 24*24 images to 12*12,
// computing the maximum value for each 2*2 input region.
void forward_maxpool_layer() {
  int i, j, k, n, m, row, col, index;
  int size=2, stride=2;
  int oh=12, ow=12;
  int ih=24, iw=24, chan=22;
  float max = -FLT_MAX, val;
  float *pout, *pin;

  pin = (float *)matCbias;
  pout = (float *)matCpool;
  
  for(k = 0; k < chan; ++k){
    for(i = 0; i < oh; ++i) {
      for(j = 0; j < ow; ++j) {
	max = -FLT_MAX;
	for(n = 0; n < size; ++n){
	  for(m = 0; m < size; ++m){
	    row = i*stride + n;
	    col = j*stride + m;
	    index = col + iw * (row + ih * k);
	    val = pin[index] ;
	    max = (val > max) ? val : max;
	  }
	}
	pout[j + ow * (i + oh * k)] = max;
      }
    }
  }
  // print_fp((float *)matCpool, 120, "Pool");
  // Output matrix Cpool is 22*144, that is this layer outputs 22 12*12 images.
}

// The convolution layer consists of 22 feature maps.
// Each feature map has a set of 5*5 weights and a single bias.
void forward_convolutional_layer() {
    // Matrix A is prepared (with 24*24=576 rows and 5*5=25 columns)
    // in order to do the convolutions as a matrix multiplication
    // such that, A(576*25) * BT(25*22) -> C(576*22)
    prepare_matrixA();
    //print_fp_mat(matA,576,25);

    // The 22 maps weights are stored as a 22*5*5 matrix (after the initial 22 bias values)
    matB = fp_weights + 22;
    
    // Matrix B is transposed to 25*22 for multiplication
    // You can do (1) transpose + gemm, or (2) gemmBT
    // transpose((float *)matB, 22, 25, (float *)matBT);
    // gemm((float *)matA, (float *)matBT, (float *)matC, 24*24, 25, 22);
    gemmBT((float *)matA, (float *)matB, (float *)matC, 24*24, 25, 22);
    
    // Add bias and transpose. 
    add_bias((float *)matC, 24*24, 22, (float *)fp_weights, (float *)matCbias, 1);
    // print_fp((float *)matCbias, 300, "Convolutional+Bias");
    // There is no activation function
    // Output matrix Cbias is 22*576, that is this layer outputs 22 24*24 images.
}

// This layer fully connects the 3168 inputs (22 12*12images),
// to 10 output neurons (one for each digit)
void forward_connected_layer() {
    float *matW, *matIN, *mbias, *matOUT, *matOutB;

    // The 10 bias values of this layer are stored after the 22+550 convolutional bias+weigths
    mbias = (float *)fp_weights + 22 + 550;
    // The 10*2880 weights are stored after the 10 bias values
    matW = (float *)fp_weights + 22 + 550 + 10;
    
    matIN = (float *)matCpool;
    matOUT = (float *)matConn;
    matOutB = (float *)matConnB;
    
    // A(10*3168) * B(3168*1) -> C(10*1)
    gemm(matW, matIN, matOUT, 10, 3168, 1);
    // print_fp((float *)matConn, 10, "Connected");
    // print_fp(mbias, 10, "Bias");
      
    add_bias(matOUT, 10, 1, mbias, (float *)matOutB, 0);
    // print_fp((float *)matConnB, 10, "Connected+Bias");
    // Output vector ConnB has 10 values, one for each digit
}

// Digit classification is performed using 4 layers:
// 1. Convolutional layer
// 2. Pooling layer
// 3. Fully-connected layer
// 4. Softmax Layer
int predict_mnist() {
  int best;
  
  forward_convolutional_layer();
  forward_maxpool_layer();
  forward_connected_layer();
  best = forward_softmax_layer();

  return best;
}

void define_memory_regions() {
  float *paddress = (float *)MEM_DATA_BASE_ADDRESS;

  // Region Size NIMAGES*IMAGE_HEIGHT*IMAGE_WIDTH+16 = 78416 Bytes (100 images)
  ch_images = (unsigned char *)MEM_IMAGES_BASE_ADDRESS;
  // Region Size TOTAL_WEIGHTS*sizeof(float) = 29330*4 = 117320 Bytes
  fp_weights = (float *)MEM_WEIGHTS_BASE_ADDRESS;

  // Region Size IMAGE_HEIGHT*IMAGE_WIDTH*sizeof(float) = 28*28*4 = 3136 Bytes
  fp_image = paddress;
  paddress += 28*28;

  // Aux matrix of (24*24)*(25) elements. Region Size = 14400 * 4 = 57600 Bytes
  matA = paddress;
  paddress += (24*24)*(25);
  // Transpose of matA. Region Size = 14400 * 4
  matAT = paddress;
  paddress += (24*24)*(25);
  // Aux matrix of (22)*(25) elements. Region Size = 550 * 4 Bytes;
  matB = paddress;
  paddress += (22)*(25);
  // Transpose of matB. Region Size = 550 * 4 Bytes
  matBT = paddress;
  paddress += (22)*(25);
  // Aux matrix of (24*24)*(22) elements. Region Size = 12672 * 4 Bytes;
  matC = paddress;
  paddress += (24*24)*(22);
  // Transpose of matC. Region Size = 11520 * 4 Bytes
  matCT = paddress;
  paddress += (24*24)*(22);
  // Aux matrix of (22)*(24*24) elements. Region Size = 11520 * 4 Bytes
  matCbias = paddress;
  paddress += (22)*(24*24);
  // Aux matrix of (22)*(12*12) elements. Region Size = 3168 * 4 Bytes
  matCpool = paddress;
  paddress += (22)*(12*12);
  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes;
  matConn = paddress;
  paddress += 10;
  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matConnB = paddress;
  paddress += 10;
  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matSoftM = paddress;
  paddress += 10;

  // printf("%p, %d\n", (void *)paddress+10, (paddress+10)-(float *)MEM_DATA_BASE_ADDRESS);
  // Total data region size is 71898 * 4 = 287,592 Bytes

}

// Only used for PC execution
void upload_images_and_weights(void *pim, void *pwe, int size_im, int size_we) {

  char header[16];
  FILE *fimages, *fweights;
  
  // Open images file
  if ((fimages = fopen(IMAGES_FILE, "r")) == NULL) {
    printf("unable to open file <t100-images-idx3-ubyte>, terminating program\n");
    exit(1);
  }

  // Open weigths file
  if ((fweights = fopen(WEIGHTS_FILE, "r")) == NULL) {
    printf("unable to open file wb.bin, terminating program\n");
    exit(1);
  }

//  // read images
  fread(pim, 1, size_im, fimages);
  printf("Upload de imagens completo\n");

  // read weights
  fread(pwe, 4, size_we, fweights);
  printf("Upload de pesos completo\n");

  fclose(fimages);
  fclose(fweights);
}

int main(int argc, char **argv) {

    int prediction;

    printf("Definindo regiões de memória\n");
    define_memory_regions();

    printf("Fazendo upload de imagens e pesos\n");
    upload_images_and_weights((void *)ch_images, (void *)fp_weights, TOTAL_IMAGES_SIZE, TOTAL_WEIGHTS);

    printf("Classificando imagens\n");
    for (unsigned int image_to_classify = 1; image_to_classify <= NIMAGES; image_to_classify++) {
      // The pixels of the input image are scaled to the [0,1[ interval
      image_scale2float((unsigned char *)ch_images, image_to_classify, (float *)fp_image);

	  #if PRINT_IMAGE
        print_pgm((unsigned char *)ch_images, image_to_classify);
	  #endif

	  // Start measuring execution time
	  CCOUNTER_RESET();

      prediction = predict_mnist();

      // Record execution time
      uint64_t execTime = CCOUNTER_CAPTURE();

      #if PRINT_RESULTS
        printf("Image %d -> Digit %d, %f certainty. \n", image_to_classify, prediction, matSoftM[prediction]*100);
      #endif

      #if PRINT_TIME
        execTime = CCOUNTER_CAPTURE() - execTime;
        printf("Prediction took %llu clock cycles. \n", execTime);
      #endif
    }

  return 0;
}
