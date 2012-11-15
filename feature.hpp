#ifndef __FEATURE_HPP_AO8VFGBZ__
#define __FEATURE_HPP_AO8VFGBZ__

#include <opencv/cv.h>
#include <boost/ptr_container/ptr_vector.hpp>

namespace retina {
namespace feature {

struct EmptyFeatureInfo { };

template <typename T, int Dim, typename InfoT>
class FeaturePOD {
protected:
    T data_[Dim];

public:
    InfoT info;
    enum { BYTES = sizeof(InfoT) + sizeof(data_) };

protected:
    size_t info_to_binary(char* buf) const {
        InfoT* pt = (InfoT*)buf;
        *pt = info;
        return sizeof(InfoT);
    }
    size_t info_from_binary(const char* buf) {
        InfoT* pt = (InfoT*)buf;
        info = *pt;
        return sizeof(InfoT);
    }
};

template <typename T, int Dim>
class FeaturePOD<T, Dim, EmptyFeatureInfo> {
protected:
    T data_[Dim];

    size_t info_to_binary(char* buf) const { return 0; }
    size_t info_from_binary(const char* buf) { return 0; }

public:
    enum { BYTES = sizeof(data_) };
};

class BaseFeature {
public:
    virtual ~BaseFeature() {};
    virtual const char* name() const = 0;
    virtual size_t bytes() const = 0;
    virtual const void* vector_buffer() const = 0;
    virtual void to_binary(char* buffer) const = 0;
    virtual void from_binary(const char* buffer) = 0;
};

template <typename T, int Dim, typename InfoT = EmptyFeatureInfo>
class Feature : public BaseFeature, public FeaturePOD<T, Dim, InfoT> {
private:
    typedef FeaturePOD<T, Dim, InfoT> Super;
public:
    typedef T ElementType;
    typedef InfoT InfoType;
    enum { DIM = Dim };
    //DECLARE_EXCEPTION_ARG0(NotSupportedException, "operation is not supported");
    //DECLARE_EXCEPTION_ARG1(ExtractionException, "exception when extracting feature: %s", std::string);

    Feature() { memset(Super::data_, 0, sizeof(Super::data_)); }

    const T& operator[](const size_t& i) const { return Super::data_[i]; };
    T& operator[](const size_t& i) { return Super::data_[i]; };

    const T& at(const size_t& i) const { return Super::data_[i]; };
    T& at(const size_t& i) { return Super::data_[i]; };

    const T* begin() const { return Super::data_; }
    T* begin() { return Super::data_; }

    const T* end() const { return Super::data_ + Dim; }
    T* end() { return Super::data_ + Dim; }

    size_t dim() const { return Dim; };
    size_t bytes() const { return Super::BYTES; };
    const void* vector_buffer() const { return Super::data_; };

    void to_binary(char* buffer) const {
        buffer += Super::info_to_binary(buffer);
        memcpy(buffer, Super::data_, sizeof(Super::data_));
    }

    void from_binary(const char* buffer) {
        buffer += Super::info_from_binary(buffer);
        memcpy(Super::data_, buffer, sizeof(Super::data_));
    }
};

class FeatureAccessor {
public:
//    DECLARE_EXCEPTION_ARG1(FeatureNotFoundException, "feature '%s' is not found", std::string);
//    DECLARE_EXCEPTION_ARG0(NotSupportedException, "operation is not supported");

    virtual ~FeatureAccessor() {}

    virtual const char* name() const = 0;
    virtual bool is_global() const = 0;
    virtual void extract(const cv::Mat& image, char* buffer) const = 0;
    virtual void extract(const cv::Mat& image, std::vector<char>& buffer) const = 0;
    virtual void extract(const cv::Mat& image, const cv::Mat& mask, char* buffer) const = 0;
    virtual void extract(const cv::Mat& image, const cv::Mat& mask, std::vector<char>& buffer) const = 0;
    virtual BaseFeature* create_from_binary(const char* buffer) = 0;
    virtual void create_multiple_from_binary(const char* buffer, boost::ptr_vector<BaseFeature>& features) = 0;
    virtual size_t bytes() const = 0;

    static FeatureAccessor& get_by_name(const std::string& name) {
        for (FeatureAccessor* i : features_)
            if (name == i->name()) return *i;
//        throw FeatureNotFoundException(name);
    }

protected:
    FeatureAccessor() { register_accessor(this); }
    FeatureAccessor(const FeatureAccessor&);

private:
    static std::vector<FeatureAccessor*> features_;
    static void register_accessor(FeatureAccessor* ptr) { features_.push_back(ptr); }
};

#define DEFINE_GLOBAL_FEATURE(fname) \
    const char* fname::NAME = #fname;   \
    namespace {\
        class fname##Accessor : public retina::feature::FeatureAccessor {   \
        public: \
            const char* name() const { return #fname; };    \
            bool is_global() const { return true; };    \
            size_t bytes() const { return fname::BYTES; };  \
            void extract(const cv::Mat& image, char* buffer) const {    \
                fname finst = fname::extract(image);    \
                finst.to_binary(buffer);    \
            }   \
            void extract(const cv::Mat& image, std::vector<char>& buffer) const {    \
                buffer.resize(fname::BYTES);   \
                extract(image, &buffer[0]); \
            }   \
            void extract(const cv::Mat& image, const cv::Mat& mask, char* buffer) const {    \
                fname finst = fname::extract(image, mask);    \
                finst.to_binary(buffer);    \
            }   \
            void extract(const cv::Mat& image, const cv::Mat& mask, std::vector<char>& buffer) const {    \
                buffer.resize(fname::BYTES);   \
                extract(image, mask, &buffer[0]); \
            }   \
            BaseFeature* create_from_binary(const char* buffer) {   \
                BaseFeature* f = new fname();   \
                f->from_binary(buffer); \
                return f;   \
            }   \
            void create_multiple_from_binary(const char* buffer, boost::ptr_vector<BaseFeature>& features) {    \
                uint32_t sz = *((const uint32_t*)buffer);   \
                buffer += 4;    \
                features.clear();    \
                for (unsigned i = 0; i < sz; i++) {    \
                    features.push_back(new fname);  \
                    features[i].from_binary(buffer);    \
                    buffer += features[i].bytes();  \
                }   \
            }   \
        };  \
        fname##Accessor _fname##Accessor_inst;  \
    };

#define DEFINE_LOCAL_FEATURE(fname) \
    const char* fname::NAME = #fname;   \
    namespace {\
        class fname##Accessor : public retina::feature::FeatureAccessor {   \
        public: \
            const char* name() const { return #fname; };    \
            bool is_global() const { return false; };    \
            size_t bytes() const { return fname::BYTES; };  \
            void extract(const cv::Mat& image, char* buffer) const {    \
                std::vector<fname> features = fname::extract(image); \
                save_to_binary(features, buffer);   \
            }   \
            void extract(const cv::Mat& image, std::vector<char>& buffer) const {    \
                std::vector<fname> features = fname::extract(image); \
                buffer.resize(features.size() * fname::BYTES + sizeof(uint32_t));   \
                save_to_binary(features, &buffer[0]);   \
            }   \
            void extract(const cv::Mat& image, const cv::Mat& mask, char* buffer) const {    \
                std::vector<fname> features = fname::extract(image, mask); \
                save_to_binary(features, buffer);   \
            }   \
            void extract(const cv::Mat& image, const cv::Mat& mask, std::vector<char>& buffer) const {    \
                std::vector<fname> features = fname::extract(image, mask); \
                buffer.resize(features.size() * fname::BYTES + sizeof(uint32_t));   \
                save_to_binary(features, &buffer[0]);   \
            }   \
            BaseFeature* create_from_binary(const char* buffer) {   \
                BaseFeature* f = new fname();   \
                f->from_binary(buffer); \
                return f;   \
            }   \
            void create_multiple_from_binary(const char* buffer, boost::ptr_vector<BaseFeature>& features) {    \
                uint32_t sz = *((const uint32_t*)buffer);   \
                buffer += 4;    \
                features.clear();    \
                for (unsigned i = 0; i < sz; i++) {    \
                    features.push_back(new fname);  \
                    features[i].from_binary(buffer);    \
                    buffer += features[i].bytes();  \
                }   \
            }   \
        private: \
            void save_to_binary(std::vector<fname>& features, char* ptr) const {  \
                *((uint32_t*)ptr) = features.size(); \
                ptr += sizeof(uint32_t);    \
                for (size_t i = 0; i < features.size(); i++) {  \
                    features[i].to_binary(ptr); \
                    ptr += fname::BYTES;   \
                }   \
            }   \
        };  \
        fname##Accessor _fname##Accessor_inst;  \
    };


} /* feature  */
} /* retina  */

#endif /* end of include guard: __FEATURE_HPP_AO8VFGBZ__ */
