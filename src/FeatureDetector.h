#pragma once
#include <GSLAM/core/KeyPoint.h>
#include <GSLAM/core/GImage.h>
#include <GSLAM/core/SPtr.h>
#include <GSLAM/core/Svar.h>

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

    static SPtr<FeatureDetector> create(std::string desireType);
};

typedef SPtr<FeatureDetector> FeatureDetectorPtr;
typedef FeatureDetectorPtr (*funcCreateFeatureDetector)();

inline FeatureDetectorPtr FeatureDetector::create(std::string desireType){
    auto& inst=GSLAM::SvarWithType<funcCreateFeatureDetector>::instance();
    funcCreateFeatureDetector createFunc=inst.get_var(desireType,NULL);
    if(!createFunc) return FeatureDetectorPtr();
    return createFunc();
}

#define REGISTER_FEATUREDETECTOR(D,E) \
    extern "C" FeatureDetectorPtr create##E(){ return FeatureDetectorPtr(new D());}\
    class D##E##_Register{ \
    public: D##E##_Register(){\
    GSLAM::SvarWithType<funcCreateFeatureDetector>::instance().insert(#E,create##E);\
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
