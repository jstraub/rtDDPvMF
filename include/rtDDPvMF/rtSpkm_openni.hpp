/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <cudaPcl/openniVisualizer.hpp>
#include <rtDDPvMF/rtSpkm.hpp>

using namespace Eigen;

//#define RM_NANS_FROM_DEPTH
// normals without nans
//TimerLog: stats over timer cycles (mean +- 3*std):  3.99052+- 6.63986 16.312+- 10.869 43.0436+- 19.7957
// full depth image with nans
//TimerLog: stats over timer cycles (mean +- 3*std):  4.58002+- 9.72736 19.0138+- 17.9823 49.9746+- 30.6944


class RealtimeSpkm : public cudaPcl::OpenniVisualizer
{
  public:
    RealtimeSpkm(shared_ptr<RtSpkm>& pRtSpkm);
    virtual ~RealtimeSpkm();

    virtual void depth_cb(const uint16_t* depth, uint32_t
        w, uint32_t h);
  protected:
    virtual void visualizeNormals();

   shared_ptr<RtSpkm> pRtSpkm_;
};
// ---------------------------------- impl -----------------------------------

RealtimeSpkm::RealtimeSpkm(shared_ptr<RtSpkm>& pRtSpkm)
  : pRtSpkm_(pRtSpkm)
{}

RealtimeSpkm::~RealtimeSpkm()
{}

void RealtimeSpkm::depth_cb(const uint16_t* depth, uint32_t w, uint32_t
    h)
{
  pRtSpkm_->compute(depth,w,h);
  this->update();
}

void RealtimeSpkm::visualizeNormals()
{
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtSpkm_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
}
