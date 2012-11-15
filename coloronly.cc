#include <opencv2/opencv.hpp>
#include "macros.h"

using namespace std;

DEFINE_string(filename, "", "file name");

extern void ColorMeanSignal(const cv::Mat& image,
                            vector<uint16_t>* color_signal);

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_filename != "");
  cv::Mat img(cv::imread(FLAGS_filename, CV_LOAD_IMAGE_COLOR));
  vector<uint16_t> color;
  ColorMeanSignal(img, &color);
  cout << ".";
  for (size_t i = 0; i < color.size(); i++) {
    cout << "\t" << color[i];
  }
  cout << endl;
}
