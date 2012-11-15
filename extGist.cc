#include "gist_feature.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using std::cout;
using retina::feature::GistFeature;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    exit(-1);
  }

  cv::Mat img(cv::imread(argv[1]));
  GistFeature feature = GistFeature::extract(img); 
  cout << ".";
  for (size_t i = 0; i < feature.dim(); i++)
    cout << "\t" << feature[i];
  cout << std::endl;
}
