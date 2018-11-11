/**
 * File: FORB.h
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_F_SIFT__
#define __D_T_F_SIFT__

#include <opencv2/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2 {

/// Functions to manipulate BRIEF descriptors
class FSift: protected FClass
{
public:

  /// Descriptor type
  typedef cv::Mat TDescriptor; // CV_32F
  /// Pointer to a single descriptor
  typedef const TDescriptor *pDescriptor;
  /// Descriptor length (in bytes)
  static const int L = 128;

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors, 
    TDescriptor &mean){
      if(descriptors.empty()) return;

      if(descriptors.size() == 1)
      {
          mean = descriptors[0]->clone();
          return;
      }
      assert(descriptors[0]->type()==CV_32F );//ensure it is float

      mean.create(1, descriptors[0]->cols,descriptors[0]->type());
      mean.setTo(cv::Scalar::all(0));
      float inv_s =1./double( descriptors.size());
      for(size_t i=0;i<descriptors.size();i++)
          mean +=  (*descriptors[i]) * inv_s;
  }
  
  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b){
      double sqd = 0.;
      assert(a.type()==CV_32F);
      assert(a.rows==1);
      const float *a_ptr=a.ptr<float>(0);
      const float *b_ptr=b.ptr<float>(0);
      for(int i = 0; i < a.cols; i ++)
          sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
      return sqd;
  }

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a)
  {
    std::stringstream ss;
    for(int i = 0; i < FSift::L; ++i)
    {
      ss << a.at<float>(i) << " ";
    }
    return ss.str();
  }
  
  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s)
  {
      a.create(1,L,CV_32F);

    std::stringstream ss(s);
    for(int i = 0; i < FSift::L; ++i)
    {
      ss >> a.at<float>(i);
    }
  }


};

} // namespace DBoW2

#endif


