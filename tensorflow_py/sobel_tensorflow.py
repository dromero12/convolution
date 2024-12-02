import sys
import cv2
import numpy as np
import tensorflow as tf


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"File not found: {image_path}")
    return image / 255.0


def create_sobel_kernel(size, direction):
    if size % 2 == 0 or size < 3:
        raise ValueError("Kernel size must be an odd number >= 3.")

    half_size = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    if direction == 'x':
        for i in range(size):
            for j in range(size):
                kernel[i, j] = (j - half_size) / ((j - half_size)**2 + (i - half_size)**2 + 1e-6)
    elif direction == 'y':
        for i in range(size):
            for j in range(size):
                kernel[i, j] = (i - half_size) / ((j - half_size)**2 + (i - half_size)**2 + 1e-6)
    else:
        raise ValueError("Direction must be 'x' or 'y'.")

    # Normalize the kernel
    kernel /= np.sum(np.abs(kernel))
    return kernel


def apply_sobel_operator(image, kernel_size=3):
    # Convert image to a 4D tensor (batch_size, height, width, channels)
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=-1)

    # Generate Sobel kernels
    sobel_x = create_sobel_kernel(kernel_size, 'x')
    sobel_y = create_sobel_kernel(kernel_size, 'y')

    # Convert kernels to TensorFlow tensors
    sobel_x_tensor = tf.convert_to_tensor(sobel_x.reshape((kernel_size, kernel_size, 1, 1)), dtype=tf.float32)
    sobel_y_tensor = tf.convert_to_tensor(sobel_y.reshape((kernel_size, kernel_size, 1, 1)), dtype=tf.float32)

    # Convolve with Sobel X and Sobel Y
    sobel_x_output = tf.nn.conv2d(input_tensor, sobel_x_tensor, strides=[1, 1, 1, 1], padding="SAME")
    sobel_y_output = tf.nn.conv2d(input_tensor, sobel_y_tensor, strides=[1, 1, 1, 1], padding="SAME")

    # Compute gradient magnitude
    gradient_magnitude = tf.sqrt(tf.square(sobel_x_output) + tf.square(sobel_y_output))

    return sobel_x_output[0, :, :, 0].numpy(), sobel_y_output[0, :, :, 0].numpy(), gradient_magnitude[0, :, :, 0].numpy()


def show_results(original_image, sobel_x, sobel_y, gradient_magnitude):
    sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    original_image = (original_image * 255).astype(np.uint8)

    # Show results
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Sobel X", sobel_x)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.imshow("Gradient Magnitude", gradient_magnitude)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_path, kernel_size):
    image = load_image(image_path)

    # Apply Sobel operator
    sobel_x, sobel_y, gradient_magnitude = apply_sobel_operator(image, kernel_size)

    show_results(image, sobel_x, sobel_y, gradient_magnitude)


if __name__ == "__main__":
    print("\n Use: python3 sobel_tensorflow.py <image_path> <kernel_size>\n")

    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
 
