#include <GSLAM/core/Svar.h>
#include <GSLAM/core/Timer.h>
#include <GSLAM/core/Glog.h>
#include <GSLAM/core/MemoryMetric.inc> // This file should always only included by the main.cpp

#include "../DBoW2/DBoW2.h"
#include "DBoW2/FSift.h"
// These undefs are required since DBoW2 and DBoW2 headers use the same names
#undef __D_T_BOW_VECTOR__
#undef __D_T_SCORING_OBJECT__
#undef __D_T_FEATURE_VECTOR__
#undef __D_T_DATABASE__
#undef __D_T_QUERY_RESULTS__
#include "../DBow3/src/DBoW3.h"
#include "../fbow/src/vocabulary_creator.h"
#include <GSLAM/core/Vocabulary.h>

#include <opencv2/highgui/highgui.hpp>
#include "FeatureDetector.h"


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
        bool mem=svar.Get<bool>("mem");
        GSLAM::ScopedTimer tm("GSLAM::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            GSLAM::Vocabulary voc(vocabularyfile2save);
            LOG_IF(INFO,mem)<<"GSLAM used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
        }

        LOG_IF(INFO,mem)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
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

void testDBoW2ORB(const std::vector<cv::Mat>& features)
{
    OrbVocabulary voc(svar.GetInt("k"),svar.GetInt("level"));
    LOG(INFO)<<"DBoW2: Creating vocabulary from image features.";
    {
        GSLAM::ScopedTimer tm("DBoW2::train");
        std::vector<std::vector<cv::Mat> > training_features;
        training_features.reserve(features.size());
        for(auto& it:features) training_features.push_back(toVec(it));
        voc.create(training_features);
    }
    LOG(INFO)<<"Created "<<voc;

    DBoW2::BowVector     v;
    DBoW2::FeatureVector fv;
    int& levels_up=svar.GetInt("levels_up");
    for(auto& it:features)
    {
        GSLAM::ScopedTimer tm("DBoW2::transImage");
        voc.transform(toVec(it),v,fv,levels_up);
    }

    {
        for(auto& it:features)
            for(int i=0;i<it.rows;i++){
                GSLAM::ScopedTimer tm("DBoW2::transDes");
                voc.transform(it.row(i));
            }
    }

    LOG(INFO)<<"DBoW2: Saving vocabulary "<<voc;
    {
        GSLAM::ScopedTimer tm("DBoW2::save");
        voc.save("vocabulary.yaml");
    }

    {
        bool mem=svar.Get<bool>("mem");
        GSLAM::ScopedTimer tm("DBoW2::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            SPtr<OrbVocabulary> voc(new OrbVocabulary(svar.GetInt("k"),svar.GetInt("level")));
            voc->load("vocabulary.yaml");
            LOG_IF(INFO,mem)<<"DBoW2 used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
        }
        LOG_IF(INFO,mem)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
    }
}

void testDBoW2Sift(const std::vector<cv::Mat>& features)
{
    typedef DBoW2::TemplatedVocabulary<DBoW2::FSift::TDescriptor, DBoW2::FSift>
      SiftVocabulary;
    SiftVocabulary voc(svar.GetInt("k"),svar.GetInt("level"));
    LOG(INFO)<<"DBoW2: Creating vocabulary from image features.";
    {
        GSLAM::ScopedTimer tm("DBoW2::train");
        std::vector<std::vector<cv::Mat> > training_features;
        training_features.reserve(features.size());
        for(auto& it:features) training_features.push_back(toVec(it));
        voc.create(training_features);
    }
    LOG(INFO)<<"Created "<<voc;

    DBoW2::BowVector     v;
    DBoW2::FeatureVector fv;
    int& levels_up=svar.GetInt("levels_up");
    for(auto& it:features)
    {
        GSLAM::ScopedTimer tm("DBoW2::transImage");
        voc.transform(toVec(it),v,fv,levels_up);
    }

    {
        for(auto& it:features)
            for(int i=0;i<it.rows;i++){
                GSLAM::ScopedTimer tm("DBoW2::transDes");
                voc.transform(it.row(i));
            }
    }

    LOG(INFO)<<"DBoW2: Saving vocabulary "<<voc;
    {
        GSLAM::ScopedTimer tm("DBoW2::save");
        voc.save("vocabulary.yaml");
    }

    {
        bool mem=svar.Get<bool>("mem");
        GSLAM::ScopedTimer tm("DBoW2::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            SPtr<SiftVocabulary> voc(new SiftVocabulary(svar.GetInt("k"),svar.GetInt("level")));
            voc->load("vocabulary.yaml");
            LOG_IF(INFO,mem)<<"DBoW2 used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
        }
        LOG_IF(INFO,mem)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
    }
}

void testDBoW2(const std::vector<cv::Mat>& features)
{
    std::cout<<"----------------DBoW2------------------------\n";
    std::string feature=svar.GetString("feature");
    if("ORB"==feature) return testDBoW2ORB(features);
    else return testDBoW2Sift(features);
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
        bool mem=svar.Get<bool>("mem");
        GSLAM::ScopedTimer tm("DBoW3::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            SPtr<DBoW3::Vocabulary> voc(new DBoW3::Vocabulary(svar.GetInt("k"),svar.GetInt("level")));
            voc->load("vocabulary.dbow");
            LOG_IF(INFO,mem)<<"DBoW3 used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
        }
        LOG_IF(INFO,mem)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
    }

}

void testFBoW(const std::vector<cv::Mat>& features)
{
    std::cout<<"----------------FBoW------------------------\n";
    fbow::Vocabulary        voc;
    LOG(INFO)<<"FBoW: Creating vocabulary from image features.";
    {
        GSLAM::ScopedTimer tm("FBoW::train");
        fbow::VocabularyCreator creator;
        creator.create(voc,features,"ORB",fbow::VocabularyCreator::Params(svar.GetInt("k"),
                                                                          svar.GetInt("level"),
                                                                          svar.GetInt("threads")));
    }
    LOG(INFO)<<"Created "<<voc.size();


    fbow::fBow     v;
    fbow::fBow2    fv;
    int level=svar.GetInt("k")-svar.GetInt("levels_up");
    for(auto& it:features)
    {
        GSLAM::ScopedTimer tm("FBoW::transImage");
        voc.transform(it,level,v,fv);
    }

    {
        for(auto& it:features)
            for(int i=0;i<it.rows;i++){
                GSLAM::ScopedTimer tm("FBoW::transDes");
                voc.transform(it.row(i));
            }
    }

    LOG(INFO)<<"FBoW: Saving vocabulary ...";
    {
        GSLAM::ScopedTimer tm("FBoW::save");
        voc.saveToFile("vocabulary.fbow");
    }

    {
        bool mem=svar.Get<bool>("mem");
        GSLAM::ScopedTimer tm("FBoW::load");
        auto before=GSLAM::MemoryMetric::instanceCPU().usage();
        auto count =GSLAM::MemoryMetric::instanceCPU().count();
        {
            SPtr<fbow::Vocabulary> voc(new fbow::Vocabulary());
            voc->readFromFile("vocabulary.fbow");
            LOG_IF(INFO,mem)<<"FBoW used memory "<<GSLAM::MemoryMetric::instanceCPU().usage()-before
                    <<" bytes with "<<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
        }
        LOG_IF(INFO,mem)<<"Memory leak:"<<GSLAM::MemoryMetric::instanceCPU().usage()-before<<" bytes with "
                <<GSLAM::MemoryMetric::instanceCPU().count()-count<<" pieces.";
    }

}

int  main(int argc,char** argv){
    svar.Arg<int>("k",10,"How many branches each node grow.");
    svar.Arg<int>("level",3,"How many levels should the vocabulary contains.");
    svar.Arg<int>("weight",3,"How many levels should the vocabulary contains.");
    svar.Arg<int>("score",3,"How many levels should the vocabulary contains.");
    svar.Arg<int>("threads",1,"How many threads use to train a model.");
    svar.Arg<std::string>("images","","The file path listed image paths.");
    svar.Arg<bool>("mem",true,"Should report memory usage or not.");
    svar.Arg<std::string>("feature","ORB","Feature name to test, support ORB or Sift.");

    auto unparsed=svar.ParseMain(argc,argv);

    std::string& image_lists=svar.GetString("images");
    if(unparsed.size()||svar.GetInt("help")||image_lists.empty())
    {
        std::cout<<svar.help()<<std::endl;
        return 0;
    }

    if(svar.Get<bool>("mem"))
    {
        GSLAM::MemoryMetric::instanceCPU().enable();
        LOG(INFO)<<"Memory analysis started.";
    }

    // extract features
    std::ifstream ifs(image_lists);
    if(!ifs.is_open()){
        LOG(ERROR)<<"Unable to open file "<<image_lists;
        return -1;
    }

    FeatureDetectorPtr feature=FeatureDetector::create(svar.GetString("feature"));
    if(!feature) {
        LOG(ERROR)<<"Please set feature with valid name!";
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
