import numpy as np
from scipy.signal import convolve2d
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

#open the image and convert it to grayscale
img = Image.open("Input_Images/Overmane.jpg")
img = img.convert('L')

kernel_size = 7
sigma = 3
#create the gaussian kernel
img_array = np.asarray(img)
def gaussian_kernel(size, sigma):
    kernel = np.empty((size, size))
    kernel_center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - kernel_center
            y = j - kernel_center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / np.sum(kernel)
    return kernel

gaussian_kernel = gaussian_kernel(kernel_size, sigma)
#convolve the gaussian kernel with the original image to reduce noise
blurred_img = convolve2d(img_array, gaussian_kernel, mode = 'same')
#we create the 2 sobel kernel that will detect the edges on the y and x axis of the image by convolving them with the image
sobel_kernelx = np.array([
    [-1 ,0 ,1],
    [-2, 0 ,2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_kernely = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2 ,1]
], dtype=np.float32)

Gx = convolve2d(blurred_img, sobel_kernelx, mode = 'same')
Gy = convolve2d(blurred_img, sobel_kernely, mode = 'same')
# we compute the gradient magnitude and the gradient orientation
gradient_magnitude = np.hypot(Gx, Gy)
gradient_direction = np.arctan2(Gy, Gx)


# we implement non-max surpression, double thresholding and edge tracking by hysteresis
def non_max_surpression_double_thresholding_track_hysteresis(magnitude, direction, high_thresh, low_thresh):
    height, width = magnitude.shape
    surpressed_img = np.zeros_like(magnitude)
    direction = np.rad2deg(direction) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = direction[i ,j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbours = (magnitude[i, j+1], magnitude[i, j-1])
            elif (22.5 <= angle < 67.5):
                neighbours = (magnitude[i-1, j+1], magnitude[i+1, j-1])
            elif (67.5 <= angle < 112.5):
                neighbours = (magnitude[i+1, j], magnitude[i-1, j])
            elif (112.5 <= angle < 157.5):
                neighbours = (magnitude[i+1, j+1], magnitude[i-1, j-1])
            if magnitude[i, j] >= max(neighbours):
                surpressed_img[i, j] = magnitude[i, j]
    edge_map = np.zeros_like(surpressed_img)
    strong_edges_row, strong_edges_col = np.where(surpressed_img >= high_thresh)
    weak_edges_row, weak_edges_col = np.where((surpressed_img < high_thresh) & (surpressed_img >= low_thresh))
    edge_map[strong_edges_row, strong_edges_col] = 255
    
    for i, j in zip(weak_edges_row, weak_edges_col):
        if (edge_map[i-1:i+2, j-1:j+2].max() == 255):
            edge_map[i, j] = 255
        else:
            edge_map[i, j] = 1

    return edge_map

high_thresh = 0.05 * gradient_magnitude.max()
low_thresh = 0.01 * high_thresh 
image = non_max_surpression_double_thresholding_track_hysteresis(gradient_magnitude, gradient_direction, high_thresh, low_thresh)
image = Image.fromarray(image.astype(np.uint8))
image.save('Output_Images/image.jpg')
