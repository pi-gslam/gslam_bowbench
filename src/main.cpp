#include <GSLAM/core/Svar.h>
#include <GSLAM/core/Timer.h>
#include <GSLAM/core/Glog.h>

#include <opencv2/highgui/highgui.hpp>

#include "../DBow3/src/DBoW3.h"
#include "../fbow/src/fbow.h"
#include <GSLAM/core/Vocabulary.h>

#include "FeatureDetector.h"
#include "MemoryMetric.h"

void testGSLAM(const std::vector<cv::Mat>& featuresCV)
{
    std::cout<<"----------------GSLAM------------------------\n";
    std::vector<GSLAM::GImage> features;
    features.reserve(featuresCV.size());
    for(auto it:featuresCV) features.push_back(it);

    LOG(INFO)<<"GSLAM: Creating vocabulary from image features.\n";
    SPtr<GSLAM::Vocabulary> vocabulary;

    {
        GSLAM::ScopedTimer tm("GSLAM::train");
        vocabulary=GSLAM::Vocabulary::create(features,svar.GetInt("k"),svar.GetInt("level"));
    }

    LOG(INFO)<<(*vocabulary)<<std::endl;

    std::string vocabularyfile2save=svar.GetString("Vocabulary.Save","vocabulary.gbow");
    if(vocabularyfile2save.size())
    {
        GSLAM::ScopedTimer timerSaveVocabulary("GSLAM::save");
        vocabulary->save(vocabularyfile2save);
    }

    if(vocabularyfile2save.size())
    {
        GSLAM::ScopedTimer tm("GSLAM::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            GSLAM::Vocabulary voc(vocabularyfile2save);
            LOG(INFO)<<"GSLAM used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" picise.";
            LOG(INFO)<<"Des:"<<voc.m_nodeDescriptors.total()<<"SizeofMat:"<<sizeof(cv::Mat)
                    <<"GImage:"<<sizeof(GSLAM::GImage)
                    <<"Node:"<<voc.m_nodes.capacity()<<"*"<<sizeof(GSLAM::Vocabulary::Node)
                   <<"Words:"<<voc.m_words.capacity()*sizeof(void*);
        }

        LOG(INFO)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" picise.";
    }

    GSLAM::BowVector     v;
    GSLAM::FeatureVector fv;
    int& levels_up=svar.GetInt("levels_up");
    for(auto& it:features)
    {
        GSLAM::ScopedTimer tm("GSLAM::transImage");
        vocabulary->transform(it,v,fv,levels_up);
    }


    {
        for(auto& it:features)
            for(int i=0;i<it.rows;i++){
                GSLAM::ScopedTimer tm("GSLAM::transDes");
                vocabulary->transform(it.row(i));
            }
    }
}

std::vector<cv::Mat> toVec(const cv::Mat& mat){
    std::vector<cv::Mat> vec;
    vec.reserve(mat.rows);
    for(int i=0;i<mat.rows;i++)
        vec.push_back(mat.row(i));
    return vec;
}

void testDBoW2(const std::vector<cv::Mat>& features)
{
    std::cout<<"----------------DBoW2------------------------\n";
}

void testDBoW3(const std::vector<cv::Mat>& features)
{
    std::cout<<"----------------DBoW3------------------------\n";
    DBoW3::Vocabulary voc(svar.GetInt("k"),svar.GetInt("level"));
    LOG(INFO)<<"DBoW3: Creating vocabulary from image features.";
    {
        GSLAM::ScopedTimer tm("DBoW3::train");
        voc.create(features);
    }
    LOG(INFO)<<"Created "<<voc;

    DBoW3::BowVector     v;
    DBoW3::FeatureVector fv;
    int& levels_up=svar.GetInt("levels_up");
    for(auto& it:features)
    {
        GSLAM::ScopedTimer tm("DBoW3::transImage");
        voc.transform(toVec(it),v,fv,levels_up);
    }

    {
        for(auto& it:features)
            for(int i=0;i<it.rows;i++){
                GSLAM::ScopedTimer tm("DBoW3::transDes");
                voc.transform(it.row(i));
            }
    }

    LOG(INFO)<<"DBoW3: Saving vocabulary "<<voc;
    {
        GSLAM::ScopedTimer tm("DBoW3::save");
        voc.save("vocabulary.dbow",false);
    }

    {
        GSLAM::ScopedTimer tm("DBoW3::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            SPtr<DBoW3::Vocabulary> voc(new DBoW3::Vocabulary(svar.GetInt("k"),svar.GetInt("level")));
            voc->load("vocabulary.dbow");
            LOG(INFO)<<"DBoW3 used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" picise.";
        }
        LOG(INFO)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" picise.";
    }

}

void testFBoW(const std::vector<cv::Mat>& features)
{
}

int main(int argc,char** argv){
    svar.Arg<int>("k",10,"How many branches each node grow.");
    svar.Arg<int>("level",3,"How many levels should the vocabulary contains.");
    svar.Arg<int>("weight",3,"How many levels should the vocabulary contains.");
    svar.Arg<int>("score",3,"How many levels should the vocabulary contains.");
    svar.Arg<std::string>("images","","The file path listed image paths.");
    svar.Arg<bool>("mem",true,"Should report memory usage or not.");
    svar.Arg<std::string>("feature","ORB","Feature name to test, support ORB or Sift.");

    if(svar.Get<bool>("mem"))
        GSLAM::MemoryMetric::instanceCPU().enable();

    auto unparsed=svar.ParseMain(argc,argv);

    std::string& image_lists=svar.GetString("images");
    if(unparsed.size()||svar.GetInt("help")||image_lists.empty())
    {
        std::cout<<svar.help()<<std::endl;
        return 0;
    }

    // extract features
    std::ifstream ifs(image_lists);
    if(!ifs.is_open()){
        LOG(ERROR)<<"Unable to open file "<<image_lists;
        return -1;
    }

    FeatureDetectorPtr feature=FeatureDetector::create(svar.GetString("feature"));
    if(!feature) {
        LOG(ERROR)<<"Please set feature name!";
        return -2;
    }

    std::vector<cv::Mat> features;
    std::string line;
    while(std::getline(ifs,line)){
        cv::Mat img=cv::imread(line);
        if(img.empty()) continue;
        std::vector<GSLAM::KeyPoint> keypoints;
        GSLAM::GImage                descriptors;
        (*feature)(img,cv::Mat(),keypoints,descriptors);
        features.push_back(descriptors);
        LOG(INFO)<<line<<" found "<<descriptors.rows<<" keypoints.";
    }

    if(features.empty()) {
        LOG(ERROR)<<"No images loaded!";
        return -3;
    }

    testGSLAM(features);
    testDBoW2(features);
    testDBoW3(features);
    testFBoW(features);

    return 0;
}
