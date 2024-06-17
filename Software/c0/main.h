#include "image.h"

#define TOTAL_WEIGHTS (22+22*5*5+10+10*22*12*12)
#define FLT_MAX 3.402823466e+38F

static unsigned char memory_images[TOTAL_IMAGES_SIZE];
static float memory_weights[TOTAL_WEIGHTS];
static float memory_data[75000];

// SSRAM pre-defined data regions
#define MEM_IMAGES_BASE_ADDRESS   memory_images//0x10000000
#define MEM_WEIGHTS_BASE_ADDRESS  memory_weights//(MEM_IMAGES_BASE_ADDRESS + TOTAL_IMAGES_SIZE)
#define MEM_DATA_BASE_ADDRESS     memory_data//(MEM_WEIGHTS_BASE_ADDRESS + TOTAL_WEIGHTS)

#define PRINT_IMAGE 0
#define PRINT_TIME 1
#define PRINT_RESULTS 1

#define IMAGES_FILE "/mnt/host/t100-images-idx3-ubyte"
#define WEIGHTS_FILE "/mnt/host/wb.bin"
