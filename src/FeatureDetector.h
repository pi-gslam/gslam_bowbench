#pragma once
#include <GSLAM/core/GSLAM.h>

class FeatureDetector
{
public:
    virtual ~FeatureDetector(){}
    virtual void operator()( const GSLAM::GImage& image, const GSLAM::GImage& mask,
                             std::vector<GSLAM::KeyPoint>& keypoints,
                             GSLAM::GImage& descriptors,
                             bool useProvidedKeypoints=false ) const{}
    virtual int descriptorSize()const{return 0;}
    virtual int descriptorType()const{return GSLAM::GImageType<>::Type;}

    static std::shared_ptr<FeatureDetector> create(std::string desireType);
    static GSLAM::Svar holder(){static GSLAM::Svar var=GSLAM::Svar::object();return var;}
};

typedef std::shared_ptr<FeatureDetector> FeatureDetectorPtr;
typedef FeatureDetectorPtr (*funcCreateFeatureDetector)();

inline FeatureDetectorPtr FeatureDetector::create(std::string desireType){
    funcCreateFeatureDetector createFunc=::FeatureDetector::holder().get<funcCreateFeatureDetector>(desireType,NULL);
    if(!createFunc) return FeatureDetectorPtr();
    return createFunc();
}

#define REGISTER_FEATUREDETECTOR(D,E) \
    extern "C" FeatureDetectorPtr create##E(){ return FeatureDetectorPtr(new D());}\
    class D##E##_Register{ \
    public: D##E##_Register(){\
    ::FeatureDetector::holder().set(#E,(funcCreateFeatureDetector)create##E);\
}}D##E##_instance;


#ifdef __OPENCV_FEATURES_2D_HPP__
class OpenCVDetector : public FeatureDetector
{
public:
    OpenCVDetector(const cv::Ptr<cv::Feature2D>& feature2d):feat2d(feature2d){}

    virtual void operator()( const GSLAM::GImage& image, const GSLAM::GImage& mask,
                             std::vector<GSLAM::KeyPoint>& keypoints,
                             GSLAM::GImage& descriptors,
                             bool useProvidedKeypoints=false ) const
    {
        cv::Mat desc;
        (*feat2d)((cv::Mat)(image),(cv::Mat)(mask),
                  *(std::vector<cv::KeyPoint>*)&keypoints,desc,
                  useProvidedKeypoints);
        descriptors=desc;
    }
    virtual int descriptorSize()const{return feat2d->descriptorSize();}
    virtual int descriptorType()const{return feat2d->descriptorType();}


    cv::Ptr<cv::Feature2D> feat2d;
};
#endif
