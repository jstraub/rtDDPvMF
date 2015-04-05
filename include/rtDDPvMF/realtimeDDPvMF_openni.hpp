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

//void RealtimeDDPvMF::normals_cb(float *d_normals, uint8_t* d_haveData, uint32_t w, uint32_t h)
//{
////  cout<<"rotating pc by"<<endl<<d_R_.get()<<endl;
////  rotatePcGPU(d_normals,d_R_.data(),w*h,3);
//  tLog_.tic(-1); // reset all timers
//
////  pddpvmf_->nextTimeStep(d_normals,w*h,3,0);
//  int32_t nComp = 0;
//  float* d_nComp = this->normalExtract->d_normalsComp(nComp);
////  cout<<"compressed to "<<nComp<<endl;
//  pddpvmf_->nextTimeStepGpu(d_nComp,nComp,3,0);
//
//  tLog_.tic(0);
//  for(uint32_t i=0; i<nIter_; ++i)
//  {
//    cout<<"@"<<i<<" :"<<endl;
//    pddpvmf_->updateLabels();
//    pddpvmf_->updateCenters();
//    if(pddpvmf_->convergedCounts(nComp/100)) break;
//  }
//  tLog_.toc(0);
//  pddpvmf_->getZfromGpu(); // cache z_ back from gpu
//  if(tLog_.startLogging()) pddpvmf_->dumpStats(fout_);
//
//  {
//    boost::mutex::scoped_lock updateLock(this->updateModelMutex);
//    if(z_.rows() != w*h) z_.resize(w*h);
//    this->normalExtract->uncompressCpu(pddpvmf_->z().data(),pddpvmf_->z().rows() ,z_.data(),z_.rows());
//    K_ = pddpvmf_->getK();
//
//    centroids_ = pddpvmf_->centroids();
//    prevCentroids_ = pddpvmf_->prevCentroids(); // get them from internal since they keep track of removed clusters
//
//    std::vector<MatrixXf> Rs(std::min(centroids_.cols(),prevCentroids_.cols()));
//    for(uint32_t k=0; k<centroids_.cols(); ++k)
//      if(k < prevCentroids_.cols())
//      {
//        Rs[k] = dplv::rotationFromAtoB<float>(prevCentroids_.col(k),centroids_.col(k));
////        cout<<"k="<<k<<endl<<Rs[k]<<endl;
//      }
//    if(Rs.size()>0)
//    {
//      deltaR_ = dplv::SO3<float>::meanRotation(Rs,pddpvmf_->counts().cast<float>(),20);
//      cout<<deltaR_<<endl;
//      R_ = deltaR_*R_;
//      pddpvmf_->rotateUninstantiated(deltaR_.transpose());
////      MatrixXf RT  =  R_.transpose();
////      d_R_.set(RT); // make async
//    }
////    cout <<"R det="<<R_.determinant()<<endl<<R_<<endl;
//  }
//
//  pddpvmf_->updateState();
//  tLog_.toc(1); // total time
//  tLog_.logCycle();
//  cout<<"---------------------------------------------------------------------------"<<endl;
//  tLog_.printStats();
//  cout<<" residual="<<residual_<<endl;
//  cout<<"---------------------------------------------------------------------------"<<endl;
//
//  {
//    boost::mutex::scoped_lock updateLock(updateModelMutex);
//    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr nDispPtr = normalExtract->normalsPc();
//    nDisp_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr( new pcl::PointCloud<pcl::PointXYZRGB>(*nDispPtr));
//    normalsImg_ = normalExtract->normalsImg();
//    this->update_ = true;
//  }
//}
//
//void RealtimeDDPvMF::visualizeNormals()
//{
//  cout<<"visualizePc"<<endl;
//  //copy again
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr nDisp(
//      new pcl::PointCloud<pcl::PointXYZRGB>(*nDisp_));
////  cv::Mat nI(nDisp->height,nDisp->width,CV_32FC3);
////  for(uint32_t i=0; i<nDisp->width; ++i)
////    for(uint32_t j=0; j<nDisp->height; ++j)
////    {
////      // nI is BGR but I want R=x G=y and B=z
////      nI.at<cv::Vec3f>(j,i)[0] = (1.0f+nDisp->points[i+j*nDisp->width].z)*0.5f; // to match pc
////      nI.at<cv::Vec3f>(j,i)[1] = (1.0f+nDisp->points[i+j*nDisp->width].y)*0.5f;
////      nI.at<cv::Vec3f>(j,i)[2] = (1.0f+nDisp->points[i+j*nDisp->width].x)*0.5f;
////      nDisp->points[i+j*nDisp->width].rgb=0;
////    }
////  cv::imshow("normals",nI);
//  cv::Mat nI(this->normalsImg_.rows,this->normalsImg_.cols,CV_8UC3);
//  cv::Mat nIRGB(this->normalsImg_.rows,this->normalsImg_.cols,CV_8UC3);
//  this->normalsImg_.convertTo(nI,CV_8UC3,127.5f,127.5f);
//  cv::cvtColor(nI,nIRGB,CV_RGB2BGR);
//  cv::imshow("normals",nIRGB);
////  cv::imshow("normals",this->normalsImg_);
////  this->pc_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(nDisp);
//
////  cout<<z_.rows()<<" "<<z_.cols()<<endl;
////  cout<<z_.transpose()<<endl;
////
//
//  uint32_t Kmax = 5;
//  uint32_t k=0;
////  cout<<" z shape "<<z_.rows()<<" "<< nDisp->width<<" " <<nDisp->height<<endl;
////  cv::Mat Iz(nDisp->height/SUBSAMPLE_STEP,nDisp->width/SUBSAMPLE_STEP,CV_8UC1);
//  zIrgb = cv::Mat(nDisp->height/SUBSAMPLE_STEP,nDisp->width/SUBSAMPLE_STEP,CV_8UC3);
//  for(uint32_t i=0; i<nDisp->width; i+=SUBSAMPLE_STEP)
//    for(uint32_t j=0; j<nDisp->height; j+=SUBSAMPLE_STEP)
//      if(nDisp->points[i+j*nDisp->width].x == nDisp->points[i+j*nDisp->width].x )
//      {
//#ifdef RM_NANS_FROM_DEPTH
//        uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
//#else
//        uint8_t idz = (static_cast<uint8_t>(z_(nDisp->width*j +i)))*255/Kmax;
//#endif
//        //            cout<<"k "<<k<<" "<< z_.rows() <<"\t"<<z_(k)<<"\t"<<int32_t(idz)<<endl;
//        zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[0] = JET_b_[idz]*255;
//        zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[1] = JET_g_[idz]*255;
//        zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[2] = JET_r_[idz]*255;
//        k++;
//      }else{
//        zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[0] = 255;
//        zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[1] = 255;
//        zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[2] = 255;
//      }
//
////  cout<<this->rgb_.rows <<" " << this->rgb_.cols<<endl;
//  if(this->rgb_.rows>1 && this->rgb_.cols >1)
//  {
//    cv::addWeighted(this->rgb_ , 0.7, zIrgb, 0.3, 0.0, Icomb);
//    cv::imshow("dbg",Icomb);
//  }else{
//    cv::imshow("dbg",zIrgb);
//  }
//
//////  uint32_t k=0, Kmax=10;
//  for(uint32_t i=0; i<nDisp->width; i+=SUBSAMPLE_STEP)
//    for(uint32_t j=0; j<nDisp->height; j+=SUBSAMPLE_STEP)
////      if(nDisp->points[i+j*nDisp->width].x == nDisp->points[i+j*nDisp->width].x )
//      if(z_(nDisp->width*j +i) <= Kmax )
//      {
////        if(z_(k) == 4294967295)
////        if(z_(i+j*nDisp->width) == 4294967295)
////          cout<<" Problem"<<endl;
//
////         k = nDisp->width*j +i;
//#ifdef RM_NANS_FROM_DEPTH
//        uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
//#else
//        uint8_t idz = (static_cast<uint8_t>(z_(nDisp->width*j +i)))*255/Kmax;
////        cout << z_(nDisp->width*j +i)<<endl;
//#endif
////        if(z_(nDisp->width*j +i)>0)
////        cout<<z_(nDisp->width*j +i)<< " " <<int(idz)<<endl;
////        n->points[k] = nDisp->points[i+j*nDisp->width];
//        nDisp->points[nDisp->width*j +i].r = static_cast<uint8_t>(floor(JET_r_[idz]*255));
//        nDisp->points[nDisp->width*j +i].g = static_cast<uint8_t>(floor(JET_g_[idz]*255));
//        nDisp->points[nDisp->width*j +i].b = static_cast<uint8_t>(floor(JET_b_[idz]*255));
//        //              n->push_back(pcl::PointXYZL());
//        //              nDisp->points[i].x,nDisp->points[i].y,nDisp->points[i].z,z_(k)));
//        k++;
//      }else{
//        nDisp->points[nDisp->width*j +i].x = 0.0;
//        nDisp->points[nDisp->width*j +i].y = 0.0;
//        nDisp->points[nDisp->width*j +i].z = 0.0;
//        nDisp->points[nDisp->width*j +i].r = 255;
//        nDisp->points[nDisp->width*j +i].g = 255;
//        nDisp->points[nDisp->width*j +i].b = 255;
//      }
//
//#ifdef USE_PCL_VIEWER                                                         
//
//  this->pc_ = nDisp;
//  if(!this->viewer_->updatePointCloud(pc_, "pc"))
//    this->viewer_->addPointCloud(pc_, "pc");
//
//  if(!updateCosy(this->viewer_,R_,"R"))
//    addCosy(this->viewer_,R_, "R");
//
////  centPc = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
////      new pcl::PointCloud<pcl::PointXYZRGB>(K_,1));
////  this->viewer_->removeAllShapes();
//  for(uint32_t k=0; k<Kmax; ++k)
//  {
//    char name[20];
//    sprintf(name,"cent%d",k);
//    this->viewer_->removeShape(std::string(name));
//  }
//  for(uint32_t k=0; k<K_; ++k)
//  {
//    uint8_t idz = (static_cast<uint8_t>(k))*255/Kmax;
//    pcl::PointXYZ pt;
//    pt.x = centroids_(0,k)*1.2;
//    pt.y = centroids_(1,k)*1.2;
//    pt.z = centroids_(2,k)*1.2;
//    double r = JET_r_[idz];
//    double g = JET_g_[idz];
//    double b = JET_b_[idz];
//    char name[20];
//    sprintf(name,"cent%d",k);
//    if(!this->viewer_->updateSphere(pt,0.1,r,g,b, std::string(name)))
//      this->viewer_->addSphere(pt,0.1,r,g,b, std::string(name));
//  }
////  centroidsPc_ = centPc;
//#endif
//}


//void RealtimeDDPvMF::visualizePc()
//{
//  // Block signals in this thread
//  sigset_t signal_set;
//  sigaddset(&signal_set, SIGINT);
//  sigaddset(&signal_set, SIGTERM);
//  sigaddset(&signal_set, SIGHUP);
//  sigaddset(&signal_set, SIGPIPE);
//  pthread_sigmask(SIG_BLOCK, &signal_set, NULL);
//
//  bool showNormals =true;
//  float scale = 2.0f;
//  // prepare visualizer named "viewer"
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (
//      new pcl::visualization::PCLVisualizer ("3D Viewer"));
//
//  //      viewer->setPointCloudRenderingProperties (
//  //          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
//  viewer->initCameraParameters ();
//  cv::namedWindow("normals");
//  //cv::namedWindow("dbg");
//  cv::namedWindow("dbgNan");
//  cv::namedWindow("rgb");
//
////  int v1(0);
////  viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
////  viewer->createViewPort (0.0, 0.0, 1., 1.0, v1);
//  viewer->setBackgroundColor (255, 255, 255);
////  viewer->setBackgroundColor (0, 0, 0, v1);
////  viewer->addText ("normals", 10, 10, "v1 text", v1);
//  viewer->addCoordinateSystem (1.0);
//
////  viewer->setPosition(0,0);
////  viewer->setSize(1000,1000);
//
////  int v2(0);
////  viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
////  viewer->setBackgroundColor (0.1, 0.1, 0.1, v2);
////  viewer->addText ("pointcloud", 10, 10, "v2 text", v2);
////  viewer->addCoordinateSystem (1.0,v2);
//
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr n;
////  pcl::PointCloud<pcl::PointXYZ>::Ptr pc;
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc;
//
//  cv::Mat zIrgb;// (nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
//  cv::Mat Icomb;// (nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
//  std::stringstream ss;
//
//  Timer t;
//  while (!viewer->wasStopped ())
//  {
////    cout<<"viewer"<<endl;
//    viewer->spinOnce (10);
//    cv::waitKey(10);
//    // break, if the last update was less than 2s ago
////    if (t.dtFromInit() > 20000.0)
////    {
////      cout<<" ending visualization - waited too long"<<endl;
////      break;
////    }
////    cout<<" after break"<<endl;
//
//    // Get lock on the boolean update and check if cloud was updated
//    boost::mutex::scoped_lock updateLock(updateModelMutex);
//    if (updateRGB_)
//    {
//      cout<<"show rgb"<<endl;
//      imshow("rgb",rgb_);
//#ifdef DUMP_FRAMES
//      ss.str(""); ss<<resultsPath_<<"rgb_"<< setw(5) << setfill('0') <<nFrame_<<".jpg";
//      cout<<"writing "<< resultsPath_<< "    "<<ss.str()<<endl;
//      cv::imwrite(ss.str(),rgb_);
//#endif
//      updateRGB_ = false;
//      t=Timer();
//    }
//    if (update)
//    {
//
////      cout<<"show pc"<<endl;
////      ss.str("residual=");
////      ss <<residual_;
////      if(!viewer->updateText(ss.str(),10,20,"residual"))
////        viewer->addText(ss.str(),10,20, "residual", v1);
//
//
//      cv::Mat nI(nDisp_.height,nDisp_.width,CV_32FC3);
//      for(uint32_t i=0; i<nDisp_.width; ++i)
//        for(uint32_t j=0; j<nDisp_.height; ++j)
//        {
//          // nI is BGR but I want R=x G=y and B=z
//          nI.at<cv::Vec3f>(j,i)[0] = (1.0f+nDisp_.points[i+j*nDisp_.width].z)*0.5f; // to match pc
//          nI.at<cv::Vec3f>(j,i)[1] = (1.0f+nDisp_.points[i+j*nDisp_.width].y)*0.5f;
//          nI.at<cv::Vec3f>(j,i)[2] = (1.0f+nDisp_.points[i+j*nDisp_.width].x)*0.5f;
//        }
//      cv::imshow("normals",nI);
//
//      uint32_t Kmax = 4;
//      uint32_t k=0;
//      cout<<" z shape "<<z_.rows()<<" "<< nDisp_.width<<" " <<nDisp_.height<<endl;
////      cv::Mat Iz(nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC1);
//      zIrgb = cv::Mat(nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
//      for(uint32_t i=0; i<nDisp_.width; i+=SUBSAMPLE_STEP)
//        for(uint32_t j=0; j<nDisp_.height; j+=SUBSAMPLE_STEP)
//          if(nDisp_.points[i+j*nDisp_.width].x == nDisp_.points[i+j*nDisp_.width].x )
//          {
//#ifdef RM_NANS_FROM_DEPTH
//            uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
//#else
//            uint8_t idz = (static_cast<uint8_t>(z_(nDisp_.width*j +i)))*255/Kmax;
//#endif
////            cout<<"k "<<k<<" "<< z_.rows() <<"\t"<<z_(k)<<"\t"<<int32_t(idz)<<endl;
//            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[0] = JET_b_[idz]*255;
//            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[1] = JET_g_[idz]*255;
//            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[2] = JET_r_[idz]*255;
//            k++;
//          }else{
//            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[0] = 255;
//            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[1] = 255;
//            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[2] = 255;
//          }
//
//      cv::addWeighted(rgb_ , 0.7, zIrgb, 0.3, 0.0, Icomb);
//      cv::imshow("dbg",Icomb);
//
//      if(showNormals){
//        n = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
//            new pcl::PointCloud<pcl::PointXYZRGB>(k,1));
//        uint32_t nPoints = k;
//        k=0;
//        for(uint32_t i=0; i<nDisp_.width; i+=SUBSAMPLE_STEP)
//          for(uint32_t j=0; j<nDisp_.height; j+=SUBSAMPLE_STEP)
//            if(nDisp_.points[i+j*nDisp_.width].x == nDisp_.points[i+j*nDisp_.width].x )
//            {
////              k = nDisp_.width*j +i;
//#ifdef RM_NANS_FROM_DEPTH
//            uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
//#else
//            uint8_t idz = (static_cast<uint8_t>(z_(nDisp_.width*j +i)))*255/Kmax;
//#endif
//              n->points[k] = nDisp_.points[i+j*nDisp_.width];
//              n->points[k].r = JET_r_[idz]*255;
//              n->points[k].g = JET_g_[idz]*255;
//              n->points[k].b = JET_b_[idz]*255;
////              n->push_back(pcl::PointXYZL());
////              nDisp_.points[i].x,nDisp_.points[i].y,nDisp_.points[i].z,z_(k)));
//              k++;
//            };
//        //pcl::transformPointCloud(*n, *n, wTk);
//
//        pc = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
//            new pcl::PointCloud<pcl::PointXYZRGB>(K_,1));
//        for(uint32_t k=0; k<K_; ++k)
//        {
//          pc->points[k].x = centroids_(0,k);
//        }
//
//      }
//
////      pc = pcl::PointCloud<pcl::PointXYZ>::Ptr(
////          new pcl::PointCloud<pcl::PointXYZ>);
////      for(uint32_t i=0; i<pc_.width; i+= 5)
////        for(uint32_t j=0; j<pc_.height; j+=5)
////          pc->points.push_back(pc_cp_->points[i+j*pc_.width]);
////      pcl::transformPointCloud(*pc, *pc , wTk);
//
////      if(!updateCosy(viewer, kRw_,"mf",2.0f))
////        addCosy(viewer,kRw_,"mf",2.0f, v1);
////
//      if(showNormals)
//        if(!viewer->updatePointCloud(n, "normals"))
//          viewer->addPointCloud(n, "normals");
////
////      if(!viewer->updatePointCloud(pc, "pc"))
////        viewer->addPointCloud(pc, "pc",v2);
//
//
////        viewer->setBackgroundColor (255, 255, 255);
//#ifdef DUMP_FRAMES
//      ss.str(""); ss<<resultsPath_<<"2Dsegmentation_"<< setw(5) << setfill('0') <<nFrame_<<".png";
//      cout<<"writing "<<ss.str()<<endl;
//      cv::imwrite(ss.str(),zIrgb);
//
//      ss.str(""); ss<<resultsPath_<<"pcNormals_"<< setw(5) << setfill('0') <<nFrame_<<".png";
//      cout<<"writing "<<ss.str()<<endl;
//      viewer->saveScreenshot(ss.str());
//#endif
//
//      // Screenshot
////      vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
////        vtkSmartPointer<vtkWindowToImageFilter>::New();
////      windowToImageFilter->SetInput(dynamic_cast<vtkWindow*>(viewer->getRenderWindow().GetPointer()));
////      windowToImageFilter->SetMagnification(3); //set the resolution of the output image (3 times the current resolution of vtk render window)
////      windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
////      windowToImageFilter->Update();
////
////      vtkSmartPointer<vtkPNGWriter> writer =
////        vtkSmartPointer<vtkPNGWriter>::New();
////      writer->SetFileName("screenshot2.png");
////      writer->SetInputConnection(windowToImageFilter->GetOutputPort());
////      writer->Write();
//
//
//      nFrame_ ++;
//      update = false;
//      t=Timer();
//    }
//    updateLock.unlock();
//  }
//}


////#define BILATERAL
//void RealtimeDDPvMF::depth2smoothXYZ(float invF, uint32_t w,uint32_t h)
//{
//  cout<<"with "<<w<<" "<<h<<endl;
//  depth2floatGPU(d_depth,a,w,h);
//
////  for(uint32_t i=0; i<3; ++i)
////  {
////    depthFilterGPU(a,w,h);
////  }
//
//  //TODO compare:
//  // now smooth the derivatives
//#ifdef BILATERAL
//  cout<<"bilateral with "<<w<<" "<<h<<endl;
//  bilateralFilterGPU(a,b,w,h,6,20.0,0.05);
//  // convert depth into x,y,z coordinates
//  depth2xyzFloatGPU(b,d_x,d_y,d_z,invF,w,h,d_xyz);
//#else
//  setConvolutionKernel(h_kernel_avg);
//  for(uint32_t i=0; i<3; ++i)
//  {
//    convolutionRowsGPU(b,a,w,h);
//    convolutionColumnsGPU(a,b,w,h);
//  }
//  // convert depth into x,y,z coordinates
//  depth2xyzFloatGPU(a,d_x,d_y,d_z,invF,w,h,d_xyz);
//#endif
//}
//

