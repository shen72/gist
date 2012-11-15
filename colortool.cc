#include <opencv2/opencv.hpp>
#include <vector>
#include <time.h>
#include <string>
#include "macros.h"

using namespace std;
using namespace cv;

DEFINE_string(input, "", "");

void ColorMeanSignal(const cv::Mat& image, vector<uint16_t>* color_signal);

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  Mat image = imread(FLAGS_input);
  vector<uint16_t> signals;
  ColorMeanSignal(image, &signals);

  Mat demo_img(128, 128, CV_8UC3);

  for (size_t i = 0; i < 128; i++) {
    uint8_t* row_ptr = demo_img.ptr(i);
    for (size_t j = 0; j < 128; j++) {
      size_t idx = (i / 32 * 4 + j / 32) * 3;
      *row_ptr++ = static_cast<uint8_t>(signals[idx++] / 256);
      *row_ptr++ = static_cast<uint8_t>(signals[idx++] / 256);
      *row_ptr++ = static_cast<uint8_t>(signals[idx++] / 256);
    }
  }
  time_t timestamp = time(0);
  string windowname(ctime(&timestamp));
  namedWindow(windowname);
  imshow(windowname, demo_img);
  waitKey();
  destroyWindow(windowname);
}
