#include <opencv2/opencv.hpp>
#include <vector>
#include "macros.h"

using namespace std;
using namespace cv;

void ColorMeanSignal(const cv::Mat& image, vector<uint16_t>* color_signal) {
  const size_t block_num = 4;
  CHECK( image.depth() == CV_8U ) << "Input Mat is not a color Image"
      << endl;
  CHECK( image.channels() == 3 ) 
      << "Invalid Channels, number of target channels is " 
      << image.channels() << endl;

  cv::Mat resized_img;
  
  cv::resize(image, resized_img, cv::Size(256, 256));
  vector<uint64_t> color_bank_(block_num * block_num * 3, 0);
  
  for (size_t i = 0; i < resized_img.rows; i++) {
    uint8_t* row_ptr = resized_img.ptr(i);
    for (size_t j = 0; j < resized_img.cols; j++) {
      size_t bin_num = (i * 4 / 256 * 4 + j * 4 / 256) * 3;
      color_bank_[bin_num] += *row_ptr;
      row_ptr++;
      color_bank_[bin_num + 1] += *row_ptr;
      row_ptr++;
      color_bank_[bin_num + 2] += *row_ptr;
      row_ptr++;      
    }
  }

  color_signal->resize(color_bank_.size());

  size_t block_size = 256 * 256 / block_num / block_num;

  for (size_t i = 0; i < color_bank_.size(); i++) {
    (*color_signal)[i] = color_bank_[i] / block_size * 256;
  }
}
