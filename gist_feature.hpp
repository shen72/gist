#ifndef __GIST_FEATURE_HPP_BGA19YVE__
#define __GIST_FEATURE_HPP_BGA19YVE__

#include "feature.hpp"

namespace retina {
namespace feature {
    
class GistFeature : public Feature<uint16_t, 512> {
public:

    static const char* NAME;
    const char* name() const { return NAME; }

    static GistFeature extract(const cv::Mat& image);
    static GistFeature extract(const cv::Mat& image, const cv::Mat& mask);

};

} /* feature  */ 
} /* retina  */ 

#endif /* end of include guard: __GIST_FEATURE_HPP_BGA19YVE__ */
