#ifndef REALTIME_MF_OPENNI_HPP_INCLUDED
#define REALTIME_MF_OPENNI_HPP_INCLUDED

#include <string>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_camera/openni_image.h>

#include <Eigen/Dense>
#include <realtimeDDPvMF.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

class RealtimeDDPvMF_openni : public RealtimeDDPvMF
{
  public:
    RealtimeDDPvMF_openni(std::string mode) : RealtimeDDPvMF(mode)
    {
      // create a new grabber for OpenNI devices
      interface_ = new pcl::OpenNIGrabber();
    };
    ~RealtimeDDPvMF_openni()
    {
      delete interface_;
    };
   
    void d_cb_ (const boost::shared_ptr<openni_wrapper::DepthImage>& d)
    {
//      cout<<"depth "<<d->getFrameID()<< " @"<<d->getTimeStamp()
//        << " size: "<<d->getWidth()<<"x"<<d->getHeight()
//        <<" focal length="<<d->getFocalLength()<<endl;
      int w=d->getWidth();
      int h=d->getHeight();
      const uint16_t* data = d->getDepthMetaData().Data();
      Matrix3d mfRd = depth_cb(data,w,h).cast<double>().transpose();
//      cout<<"mfRd:"<<endl<<mfRd<<endl;
    };

  void rgb_cb_ (const boost::shared_ptr<openni_wrapper::Image>& rgb)
    {
//      cout<<"rgb "<<rgb->getFrameID()<< " @"<<rgb->getTimeStamp()
//        << " size: "<<rgb->getWidth()<<"x"<<rgb->getHeight()<<" px format:"
//        << rgb->getMetaData().PixelFormat()<<endl;
      int w = rgb->getWidth(); 
      int h = rgb->getHeight(); 
      //const uint8_t *data = rgb->getMetaData().Data(); // YUV 422
      //const uint8_t *data = rgb->getMetaData().Data();// .Grayscale8Data();

      // TODO: uggly .. but avoids double copy of the image.
      boost::mutex::scoped_lock updateLock(updateModelMutex);
      if(rgb_.cols <w)
      {
        rgb_ = cv::Mat(h,w,CV_8UC3);
      }
      //rgb_ = cv::Mat(h,w,CV_8UC3,(void*)(const uint8_t*)data);
      rgb->fillRGB(w,h,rgb_.data);
      updateRGB_ = true;
      updateLock.unlock();
    }

  protected:
    void run_impl ()
    {
      boost::function<void (const boost::shared_ptr<openni_wrapper::DepthImage>&)>
        f_d = boost::bind (&RealtimeDDPvMF_openni::d_cb_, this, _1);
      boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&)> 
        f_rgb = boost::bind (&RealtimeDDPvMF_openni::rgb_cb_, this, _1);
      // connect callback function for desired signal. 
      boost::signals2::connection c_d = interface_->registerCallback (f_d);
      boost::signals2::connection c_rgb = interface_->registerCallback (f_rgb);
      // start receiving point clouds
      interface_->start ();
    }
    void run_cleanup_impl()
    {
      // stop the grabber
      interface_->stop ();
    }
  private:
    pcl::Grabber* interface_;
};

#endif
