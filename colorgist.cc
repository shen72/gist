#include "colorgist.h"

#include <vector>

#include "gist_feature.hpp"
#include "macros.h"

using cv::Mat;
using retina::feature::GistFeature;
using namespace std;

void ColorMeanSignal(const cv::Mat& image, vector<uint16_t>* color_signal);

using namespace retina::feature;
DEFINE_GLOBAL_FEATURE(ColorGistFeature);

ColorGistFeature ColorGistFeature::extract(const Mat& image) {
  Mat image8;
  if (image.depth() != CV_8U) {
    image.convertTo(image8, CV_8U);
  } else { image8 = image; }

  vector<uint16_t> color_signal;
  ColorMeanSignal(image, &color_signal);
  CHECK(color_signal.size() == COLOR_DIM) << "Dimension check failed: "
      << "color signal deprecated." << endl;

  GistFeature rawgist512 = GistFeature::extract(image8);
  CHECK(rawgist512.DIM == GIST_DIM) << "Dimension check failed: "
      << "Gist signal deprecated." << endl;

  ColorGistFeature retval;
  for (size_t i = 0; i < color_signal.size(); i++) {
    retval[i] = color_signal[i];
  }

  for (size_t i = 0; i < GIST_DIM; i++) {
    retval[i + COLOR_DIM] = rawgist512[i];
  }
  
  return retval;
}

ColorGistFeature ColorGistFeature::extract(const Mat& image,
                                           const Mat& mask) {
  return ColorGistFeature::extract(image);
}
