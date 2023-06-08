import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from scipy.signal import convolve2d
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

gaussian_kernel = SourceModule(
"""
#include <math.h>
#define M_PI 3.14159265358979323846
 __global__ void gaussian_kernel(double *kernel, int size, double sigma)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int kernel_center = size / 2;

    if (i < size && j < size && i > 0 && j > 0) {
        double x = i - kernel_center;
        double y = j - kernel_center;
        double exponent = -(x * x + y * y) / (2 * sigma * sigma);
        kernel[i * size + j] = exp(exponent) / (2 * M_PI * sigma * sigma);
    }

    __syncthreads();

    if (i == kernel_center && j == kernel_center) {
        double sum = 0;
        for (int k = 0; k < size * size; k++) {
            sum += kernel[k];
        }
        for (int k = 0; k < size * size; k++) {
            kernel[k] /= sum;
        }
    }
}
"""
).get_function("gaussian_kernel")

matrix_convolution = SourceModule(
"""
__global__ void convolution2D(double *matrix, double *result, const double *mask, int M, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int mask_offset = M / 2;
    
    double temp;

    if (row < height && col < width){
        temp = 0;
        int start_r = row - mask_offset;
        int start_c = col - mask_offset;
        
        for (int i = 0; i < M; i++){
            for (int j = 0; j < M; j++){
                int r = start_r + i;
                int c = start_c + j;                
                if ( c < width  && c >= 0 && r < height  && r >= 0){
                    temp += matrix[r * width + c] * mask[i * M + j];
                }
                
            }
        }
        result[row * width + col] = temp;
    }
        
}
"""
).get_function("convolution2D")

gradient_magnitude = SourceModule(
"""
#include <math.h>
__global__ void gradientMagnitude(double *sobelx_matrix, double *sobely_matrix, double *result, int width, int height){
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     double sum;
     if (row < height && col < width){
        sum = sobelx_matrix[row * width + col] * sobelx_matrix[row * width + col] + sobely_matrix[row * width + col] * sobely_matrix[row * width + col];
        result[row * width + col] = sqrt(sum);
     }
}
"""
).get_function("gradientMagnitude")

gradient_direction = SourceModule(
"""
#include <math.h>
__global__ void gradientDirection(double *sobelx_matrix, double *sobely_matrix, double *result, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double direction;
    if (row < height && col < width){
        direction = atan2( sobely_matrix[row * width + col], sobelx_matrix[row * width + col]);
        result[row * width + col] = direction;
    }
}
"""
).get_function("gradientDirection")

non_max_surpression = SourceModule(
    """
#include <math.h>
__global__ void nonMaxSurpression(double* direction_matrix, double* magnitude_matrix, double* result, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double angle;
    
    if (row < height && col < width){

        int neighbourRow1 = row;
        int neighbourCol1 = col;
        int neighbourRow2 = row;
        int neighbourCol2 = col;

        angle = direction_matrix[row * width + col] * (180.0 / M_PI);
        if (angle < 0.0)
            angle += 180.0;
        else if (angle >= 180.0)
            angle -= 180.0;
            
        if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)){
            neighbourCol1 += 1;
            neighbourCol2 -= 1;
        }
        else if (22.5 <= angle && angle < 67.5){
            neighbourRow1 -= 1;
            neighbourCol1 += 1;
            neighbourRow2 += 1;
            neighbourCol2 -= 1;
        }
        else if (67.5 <= angle && angle < 112.5){
            neighbourRow1 += 1;
            neighbourRow2 -= 1;
        }
        else if (112.5 <= angle && angle < 157.5){
            neighbourRow1 += 1;
            neighbourCol1 += 1;
            neighbourRow2 -= 1;
            neighbourCol2 -= 1;
        }
        if (neighbourRow1 >= 0 && neighbourRow1 < height && neighbourCol1 >= 0 && neighbourCol1 < width && neighbourRow2 >= 0 && neighbourRow2 < height && neighbourCol2 >= 0 && neighbourCol2 < width) {

            if (magnitude_matrix[row * width + col] >= magnitude_matrix[neighbourRow1 * width + neighbourCol1] && magnitude_matrix[row * width + col] >= magnitude_matrix[neighbourRow2 * width + neighbourCol2]){
                result[row * width + col] = magnitude_matrix[row * width + col];    
            }
        }
    }
}
    """).get_function("nonMaxSurpression")
edge_tracking_by_hysteresis = SourceModule(
    """
#include <math.h>
__global__ void EdgeTracking(double* matrix, double* result, double high_threshold, double low_threshold, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width){
        if (matrix[row * width + col] >= high_threshold){
            result[row * width + col] = 255;
        }
        else if (matrix[row * width + col] >= low_threshold && matrix[row * width + col] < high_threshold) {
            // Check if any of the neighboring pixels are strong edges
            bool hasStrongEdgeNeighbor = false;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int neighborRow = row + i;
                    int neighborCol = col + j;

                    // Check if the neighbor is within bounds
                    if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                        // Check if the neighbor is a strong edge
                        if (matrix[neighborRow * width + neighborCol] >= high_threshold) {
                            hasStrongEdgeNeighbor = true;
                            break;
                        }
                    }
                }
                if (hasStrongEdgeNeighbor) {
                    break;
                }
            }
            result[row * width + col] = hasStrongEdgeNeighbor ? 255 : 0;
        }
        else{
            result[row * width + col] = 0;
        }
    }
}

    """
).get_function("EdgeTracking")

img = Image.open("Input_Images/Overmane.jpg")
img = img.convert('L')
img_array = np.asarray(img, dtype = np.float64)
    
sobel_kernelx = np.array([
    [1 ,0 ,-1],
    [2, 0 ,-2],
    [1, 0, -1]
], dtype=np.float64)

sobel_kernely = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2 ,-1]
], dtype=np.float64)

img_width = img_array.shape[1]
img_height = img_array.shape[0]

block_size = (16, 16, 1)
grid_size = (((img_width + block_size[0] - 1) // block_size[0]), ((img_height + block_size[1] - 1) // block_size[1]), 1)

gaussian_kernel_size = 7
gaussian_kernel_sigma = 3

img_gpu = gpuarray.to_gpu(img_array)



gaussian_kernel_gpu = gpuarray.zeros((gaussian_kernel_size, gaussian_kernel_size), dtype = np.float64)
gaussian_kernel(gaussian_kernel_gpu, np.int32(gaussian_kernel_size), np.float64(gaussian_kernel_sigma), block=block_size, grid=grid_size)
blurred_gpu = gpuarray.empty((img_height, img_width), dtype = np.float64)

matrix_convolution(img_gpu, blurred_gpu, gaussian_kernel_gpu, np.int32(gaussian_kernel_size), np.int32(img_array.shape[1]), np.int32(img_array.shape[0]), block=block_size, grid=grid_size)

edge_x_gpu = gpuarray.empty((img_height, img_width), dtype = np.float64)
sobel_kernelx_gpu = gpuarray.to_gpu(sobel_kernelx)

matrix_convolution(blurred_gpu, edge_x_gpu, sobel_kernelx_gpu, np.int32(sobel_kernelx.shape[1]), np.int32(img_width), np.int32(img_height), block=block_size, grid=grid_size)

edge_y_gpu = gpuarray.empty((img_height, img_width), dtype = np.float64)
sobel_kernely_gpu = gpuarray.to_gpu(sobel_kernely)

matrix_convolution(blurred_gpu, edge_y_gpu, sobel_kernely_gpu, np.int32(sobel_kernely.shape[1]),np.int32(img_width), np.int32(img_height), block=block_size, grid=grid_size)

G_magnitude_gpu = gpuarray.empty((img_height, img_width), dtype = np.float64)

gradient_magnitude(edge_x_gpu, edge_y_gpu, G_magnitude_gpu, np.int32(img_width), np.int32(img_height), block=block_size, grid=grid_size)

G_direction_gpu = gpuarray.empty((img_height, img_width), dtype = np.float64)

gradient_direction(edge_x_gpu, edge_y_gpu, G_direction_gpu, np.int32(img_width), np.int32(img_height), block=block_size, grid=grid_size)

surpressed_matrix_gpu = gpuarray.zeros((img_height, img_width), dtype = np.float64)
G_magnitude_cpu = G_magnitude_gpu.get()
non_max_surpression(G_direction_gpu, G_magnitude_gpu, surpressed_matrix_gpu, np.int32(img_width), np.int32(img_height), block=block_size, grid=grid_size)

result_gpu = gpuarray.empty((img_height, img_width), dtype = np.float64)


high_threshold = 0.05 * G_magnitude_cpu.max()
low_threshold = 0.01 * high_threshold

edge_tracking_by_hysteresis(surpressed_matrix_gpu, result_gpu, np.float64(high_threshold), np.float64(low_threshold), np.int32(img_width), np.int32(img_height), block=block_size, grid=grid_size)
#pizda matii
edge_result = result_gpu.get()
edge_result = Image.fromarray(edge_result.astype(np.uint8))
edge_result.save("Output_Images/imageCUDA.jpg")
  