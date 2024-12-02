#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>

using namespace cv;
using namespace std;

// Convolution function
void convolve(const Mat& input, Mat& output, const float kernel[3][3])
{  
  output = Mat::zeros(input.size(), CV_32F); // Store gradient values

  int kernelSize = 3;
  int border = kernelSize / 2;
 
  // Outer loop
  #pragma omp parallel for
  for (int y = border; y < input.rows - border; ++y)
    {
      for (int x = border; x < input.cols - border; ++x)
	{
	  float sum = 0.0;

	  // Perform the convolution
	  for (int ky = -border; ky <= border; ++ky)
	    {
	      for (int kx = -border; kx <= border; ++kx)
		{
		  int pixelVal = input.at<uchar>(y + ky, x + kx);
		  sum += pixelVal * kernel[ky + border][kx + border];
                }
            }
	  output.at<float>(y, x) = sum;
        }
    }
}


int main(int argc, char** argv)
{ 
  if (argc != 2)
    {
      cout << "Usage: " << argv[0] << " <image_file>" << endl;
      return -1;
    }

  Mat image = imread(argv[1], IMREAD_GRAYSCALE);

  if (image.empty())
    {
      cout << "Error: Could not open or find the image." << endl;
      return -1;
    }

  // Sobel kernels
  float sobelX[3][3] =
      {
       {-1, 0, 1},
       {-2, 0, 2},
       {-1, 0, 1}
      };

  float sobelY[3][3] =
    {
     {-1, -2, -1},
     { 0,  0,  0},
     { 1,  2,  1}
    };

  Mat gradientX, gradientY, gradientMagnitude;

  // Perform convolution with Sobel X and Y kernels
  double start = omp_get_wtime();
  convolve(image, gradientX, sobelX);
  convolve(image, gradientY, sobelY);
  double end = omp_get_wtime();

  cout << "Convolution completed in " << (end - start) << " seconds." << endl;

  // Calculate the gradient magnitude
  gradientMagnitude = Mat::zeros(image.size(), CV_32F);

  int x = 0, y = 0;
  float gx = 0.0, gy = 0.0;
  
  #pragma omp parallel for
  for (y = 0; y < image.rows; ++y)
    {
      for (x = 0; x < image.cols; ++x)
	{
	  gx = gradientX.at<float>(y, x);
	  gy = gradientY.at<float>(y, x);
	  gradientMagnitude.at<float>(y, x) = sqrt(gx * gx + gy * gy);
        }
    }

  // Normalize and convert gradient magnitude to displayable format
  Mat displayGradientX;
  Mat displayGradientY;
  Mat displayGradient;
  
  normalize(gradientMagnitude, displayGradient, 0, 255, NORM_MINMAX, CV_8U);
  normalize(gradientX, displayGradientX, 0, 255, NORM_MINMAX, CV_8U);
  normalize(gradientY, displayGradientY, 0, 255, NORM_MINMAX, CV_8U);

  // Display the results
  imshow("Original Image", image);
  imshow("Gradient X", displayGradientX);
  imshow("Gradient Y", displayGradientY);  
  imshow("Gradient Magnitude", displayGradient);
    
  waitKey(0);

  return 0;
}
