#include <vector>

#ifdef HAS_OPENCV
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <GSLAM/core/Glog.h>
#include <GSLAM/core/Timer.h>
#include "OpenGL.h"
#include "../FeatureDetector.h"
#include "SiftGPU/SiftGPU.h"

typedef unsigned char u_char;

using namespace std;

class GPUSIFT : public FeatureDetector
{
public:
    GPUSIFT(int nFeature = svar.GetInt("SLAM.nFeature",1000),
            int nOctaveLayers = 3,double contrastThreshold = 0.03,
            double edgeThreshold = 20,double sigma = 1.6,int verbose = 0);

    virtual ~GPUSIFT();


    virtual void operator()(const GSLAM::GImage& image, const GSLAM::GImage& mask,
                             std::vector<GSLAM::KeyPoint>& keypoints,
                             GSLAM::GImage& descriptors,
                             bool useProvidedKeypoints=false ) const;

    void    init()const;


    virtual int descriptorSize()const{return 128;}//4*128
    virtual int descriptorType()const{return GSLAM::GImageType<float>::Type;}

private:
    mutable SPtr<SiftGPU>  sift;
    mutable vector<string> args;
};


GPUSIFT::GPUSIFT(int nFeature,int nOctaveLayers,double contrastThreshold,
                 double edgeThreshold,double sigma,int verbose)
{
    // prepare arguments
    args.push_back("-tc");
    args.push_back(to_string(nFeature));
    args.push_back("-v");
    args.push_back(to_string(verbose));
    args.push_back("-e");
    args.push_back(to_string(svar.GetDouble("edge_threshold",10.0)));
    args.push_back("-d");
    args.push_back(to_string(svar.GetInt("nOctaveLayers",3)));
    args.push_back("-t");
    args.push_back(to_string(0.02/svar.GetInt("nOctaveLayers",3)));
//    args.push_back("-maxd");
//    args.push_back(to_string(svar.GetInt("GPUSIFT.max_image_size",3200)));
//    args.push_back("-tc2");
//    args.push_back(to_string(svar.GetInt("GPUSIFT.max_num_features",8192)));

    //svar.dumpAllVars();

}

GPUSIFT::~GPUSIFT()
{
}

void GPUSIFT::init()const
{
    if(!args.size()) return;

    // create SIFT object
    sift=SPtr<SiftGPU>(CreateNewSiftGPU(0));

    if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    {
        LOG(ERROR)<<"SiftGPU not full supported!";
        return;
    }


    int argc = args.size();
    char** argv = (char**) malloc(sizeof(char*)*argc);
    for(int i=0; i<argc; i++)
        argv[i] = (char*) args[i].c_str();

    sift->ParseParam(argc, argv);

    free(argv);
    args.clear();
}

void cvtColor(const GSLAM::GImage& image,GSLAM::GImage& out)
{

#ifdef HAS_OPENCV
    {
        cv::Mat imgGray;
        if( image.type() == CV_8UC3 )
            cv::cvtColor(cv::Mat(image), imgGray, cv::COLOR_BGR2GRAY);
        else if(image.type()==CV_8UC1)
            imgGray = image;
        else if(image.type()==CV_8UC4)
            cv::cvtColor(cv::Mat(image),imgGray,CV_RGBA2GRAY);
        out=imgGray;
        return ;
    }
#endif

    if( image.type() == GSLAM::GImageType<u_char,3>::Type )
    {
        out=GSLAM::GImage(image.rows,image.cols,GSLAM::GImageType<>::Type);
        for(int i=0,iend=image.total();i<iend;i++)
            out.data[i]=image.data[i*3+1];
        return ;
    }

    if( image.type() == GSLAM::GImageType<u_char,4>::Type )
    {
        out=GSLAM::GImage(image.rows,image.cols,GSLAM::GImageType<>::Type);
        for(int i=0,iend=image.total();i<iend;i++)
            out.data[i]=image.data[i*4+1];
        return;
    }

    out=image;
}

void GPUSIFT::operator()(const GSLAM::GImage& image, const GSLAM::GImage& mask,
                         std::vector<GSLAM::KeyPoint>& keypoints,
                         GSLAM::GImage& descriptors,
                         bool useProvidedKeypoints) const
{
    if( !sift ) init();
    if( !sift ) return;
    if(image.empty()) return;

    GSLAM::GImage imgGray;
    cvtColor(image,imgGray);
    if(imgGray.type()!=GSLAM::GImageType<>::Type)
    {
        LOG(ERROR)<<"SiftGPU need gray image!";
        return;
    }
    sift->RunSIFT(imgGray.cols, imgGray.rows, imgGray.data,
                  GL_LUMINANCE, GL_UNSIGNED_BYTE);// FIXME: Segment Fault running rtmv datasets

    int nFea = sift->GetFeatureNum();

    vector<SiftGPU::SiftKeypoint> kps(nFea);
    keypoints.resize(nFea);
    descriptors=GSLAM::GImage(nFea,128,GSLAM::GImageType<float>::Type,nullptr,false,32);


    sift->GetFeatureVector(&kps[0], descriptors.ptr<float>(0));

    for(int i=0; i<nFea; i++)
    {
        keypoints[i].pt.x   = kps[i].x;
        keypoints[i].pt.y   = kps[i].y;
        keypoints[i].size   = kps[i].s;
        keypoints[i].angle  = kps[i].o*180.0/3.1415926;
    }
}

REGISTER_FEATUREDETECTOR(GPUSIFT,Sift);
