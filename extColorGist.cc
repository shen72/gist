#include <opencv2/opencv.hpp>
#include <iostream>

#include "colorgist.h"
#include "macros.h"

DEFINE_string(in, "", "input image");
DEFINE_string(out, "", "output image");


using namespace std;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  cv::Mat img(cv::imread(argv[1]));
  
  ColorGistFeature feature = ColorGistFeature::extract(img); 

  cout << ".";
  for (size_t i = 0; i < feature.dim(); i++)
    cout << "\t" << feature[i];
  cout << std::endl;
}
