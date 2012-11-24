#ifndef COLORGIST_H
#define COLORGIST_H 

#include "feature.hpp"

using retina::feature::Feature;
using cv::Mat;

class ColorGistFeature : public Feature<uint16_t, 560> {
 public:
  static const char* NAME;
  const char* name() const { return NAME; }

  static ColorGistFeature extract(const Mat& image);
  static ColorGistFeature extract(const Mat& image, const Mat& mask);

  enum {
    COLOR_DIM = 48,
    GIST_DIM = 512
  };
};
#endif /* COLORGIST_H */
