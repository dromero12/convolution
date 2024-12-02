#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Convolution function
void convolve(const Mat& input, Mat& output, const float kernel[3][3]) {
    output = Mat::zeros(input.size(), input.type());

    int kernelSize = 3;
    int border = kernelSize / 2;

    for (int y = border; y < input.rows - border; ++y) {
        for (int x = border; x < input.cols - border; ++x) {
            float sum = 0.0;

            // Perform convolution
            for (int ky = -border; ky <= border; ++ky) {
                for (int kx = -border; kx <= border; ++kx) {
                    int pixelVal = input.at<uchar>(y + ky, x + kx);
                    sum += pixelVal * kernel[ky + border][kx + border];
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }
}


// Simple program to explain the convolution process (Document: Section 1)
// (single kernel based image smoothing)

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_file>" << endl;
        return -1;
    }

    // Load the image in grayscale
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error: Could not open or find the image." << endl;
        return -1;
    }

    // Smoothing kernel (for 3x3 Gaussian Blur kernel)
    float kernel[3][3] = {
        {1.0f / 16, 2.0f / 16, 1.0f / 16},
        {2.0f / 16, 4.0f / 16, 2.0f / 16},
        {1.0f / 16, 2.0f / 16, 1.0f / 16}
    };

    Mat smoothedImage;
    convolve(image, smoothedImage, kernel);

    // Display the results
    imshow("Original Image", image);
    imshow("Smoothed Image", smoothedImage);

    waitKey(0);

    return 0;
}
