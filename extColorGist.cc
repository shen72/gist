#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "colorgist.h"
#include "macros.h"

DEFINE_string(in, "", "input image");
DEFINE_string(out, "/dev/stdout", "output image");

using namespace std;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  cv::Mat img(cv::imread(FLAGS_in));
  
  ColorGistFeature feature = ColorGistFeature::extract(img); 

  ofstream fout(FLAGS_out);

  CHECK(fout) << "Unable to open " << FLAGS_out << endl;

  fout << ".";
  for (size_t i = 0; i < feature.dim(); i++)
    fout << "\t" << feature[i];
  fout << endl;
  fout.close();
}
