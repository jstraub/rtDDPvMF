/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <cudaPcl/openniVisualizer.hpp>
#include <rtDDPvMF/rtDDPvMF.hpp>

using namespace Eigen;

class RealtimeDDPvMF : public cudaPcl::OpenniVisualizer
{
  public:
    RealtimeDDPvMF(shared_ptr<RtDDPvMF>& pRtDdpvMF);
    virtual ~RealtimeDDPvMF();

    virtual void depth_cb(const uint16_t* depth, uint32_t
        w, uint32_t h);

  protected:
    virtual void visualizeNormals();

   shared_ptr<RtDDPvMF> pRtDdpvMF_;

};
// ---------------------------------- impl -----------------------------------


RealtimeDDPvMF::RealtimeDDPvMF(shared_ptr<RtDDPvMF>& pRtDdpvMF)
  : pRtDdpvMF_(pRtDdpvMF)
{}

RealtimeDDPvMF::~RealtimeDDPvMF()
{}

void RealtimeDDPvMF::depth_cb(const uint16_t* depth, uint32_t w,
    uint32_t h)
{
  Matrix3f dR = pRtDdpvMF_->applyConstVelModel();
  pRtDdpvMF_->compute(depth,w,h);
  this->update();
}

void RealtimeDDPvMF::visualizeNormals()
{
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtDdpvMF_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
}

