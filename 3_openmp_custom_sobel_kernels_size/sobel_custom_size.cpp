#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>

using namespace cv;
using namespace std;

//Create sobel kernels from user-defined size
vector<vector<float>> createSobelKernel(int kernelSize, char direction) 
  {
    if (kernelSize % 2 == 0 || kernelSize < 3)
      throw invalid_argument("Kernel size must be an odd number >= 3");
  
    int halfSize = kernelSize / 2;
    
    vector<vector<float>> kernel(kernelSize, vector<float>(kernelSize, 0));

    for (int y = -halfSize; y <= halfSize; ++y)
      {
        for (int x = -halfSize; x <= halfSize; ++x)
	  {
            if (direction == 'x') 
	      kernel[y + halfSize][x + halfSize] = x / (float)((x * x + y * y) ? (x * x + y * y) : 1);

            else if (direction == 'y')
	      kernel[y + halfSize][x + halfSize] = y / (float)((x * x + y * y) ? (x * x + y * y) : 1);

	    else
	      throw invalid_argument("Direction must be 'x' or 'y'"); 
	  }
      }

    // Kernel normalization for output gradients
    float sum = 0.0;
    for (const auto& row : kernel)
      {
        for (float val : row) 
	  sum += abs(val); 
      }

    if (sum > 0)
      {
        for (auto& row : kernel)
	  {
            for (float& val : row) 
	      val /= sum;
            
	  }
      }

    return kernel;
  }



void convolve(const Mat& input, Mat& output, const vector<vector<float>>& kernel)
{
  int kernelSize = kernel.size();
  int border = kernelSize / 2;
  
  output = Mat::zeros(input.size(), CV_32F); // Store gradient values
    
  #pragma omp parallel for
  for (int y = border; y < input.rows - border; ++y)
    {
      for (int x = border; x < input.cols - border; ++x)
	{
	  float sum = 0.0;

	  // Perform convolution
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
  if (argc != 3)
    {
      cout << "Usage: " << argv[0] << " <image_file> <kernel_size>" << endl;
      return -1;
    }

  // Load the image in grayscale
  Mat image = imread(argv[1], IMREAD_GRAYSCALE);
  if (image.empty())
    {
      cout << "Error: Could not open or find the image." << endl;
      return -1;
    }

  // Parse the kernel size
  int kernelSize = atoi(argv[2]);
  if (kernelSize % 2 == 0 || kernelSize < 3)
    {
      cout << "Error: Kernel size must be an odd number greater than or equal to 3." << endl;
      return -1;
    }

  // Create Sobel kernels dynamically based on the kernel size
  vector<vector<float>> sobelX, sobelY;
  try
    {
      sobelX = createSobelKernel(kernelSize, 'x');
      sobelY = createSobelKernel(kernelSize, 'y');
    } catch (const exception& e)
    {
      cerr << "Error: " << e.what() << endl;
      return -1;
    }

  Mat gradientX, gradientY, gradientMagnitude;

  // Perform convolution with Sobel X and Y kernels
  double start = omp_get_wtime();
  convolve(image, gradientX, sobelX);
  convolve(image, gradientY, sobelY);
  double end = omp_get_wtime();

  cout << "Convolution completed in " << (end - start) << " seconds." << endl;

  // Calculate the gradient magnitude
  gradientMagnitude = Mat::zeros(image.size(), CV_32F);

  #pragma omp parallel for
  for (int y = 0; y < image.rows; ++y)
    {
      for (int x = 0; x < image.cols; ++x)
	{
	  float gx = gradientX.at<float>(y, x);
	  float gy = gradientY.at<float>(y, x);
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
