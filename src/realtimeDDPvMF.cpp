#include <realtimeDDPvMF.hpp>


RealtimeDDPvMF::~RealtimeDDPvMF()
{
  if(!cuda_ready) return;

  checkCudaErrors(cudaFree(d_depth));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_n));
  checkCudaErrors(cudaFree(d_xyz));
  checkCudaErrors(cudaFree(d_xu));
  checkCudaErrors(cudaFree(d_yu));
  checkCudaErrors(cudaFree(d_zu));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
  checkCudaErrors(cudaFree(d_zv));
  checkCudaErrors(cudaFree(a));
  checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFree(c));
#ifdef WEIGHTED
  checkCudaErrors(cudaFree(d_weights));
#endif

  free(h_n);
  delete pddpvmf_;
  fout_.close();
}

#define SMOOTH_DEPTH

void RealtimeDDPvMF::extractNormals(const uint16_t *data, uint32_t w, uint32_t h)
{
  checkCudaErrors(cudaMemcpy(d_depth, data, w * h * sizeof(uint16_t),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());

#ifdef SMOOTH_DEPTH
  depth2smoothXYZ(invF_,w,h); 
#else
  depth2xyzGPU(d_depth,d_x,d_y,d_z,invF_,w,h,d_xyz); 
#endif
  // obtain derivatives using sobel 
  computeDerivatives(w,h);
#ifndef SMOOTH_DEPTH
  // now smooth the derivatives
  smoothDerivatives(2,w,w);
#endif
  // obtain the normals using mainly cross product on the derivatives
//  derivatives2normalsGPU(
  derivatives2normalsCleanerGPU(
      d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);


  checkCudaErrors(cudaMemcpy(h_n, d_n, w*h* X_STEP *sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
};

Matrix3f RealtimeDDPvMF::depth_cb(const uint16_t *data, int w, int h) 
{

  prepareCUDA(w,h);

  Timer t0;
  tLog_.tic(-1); // reset all timers

  extractNormals(data,w,h);

  tLog_.toc(0); 

  cout<<"w="<<w<<" " <<" h="<<h<<" "<<X_STEP<<endl;

  uint32_t nNormals = 0;
  for(uint32_t i=0; i<n_.width; i+=SUBSAMPLE_STEP)
    for(uint32_t j=0; j<n_.height; j+=SUBSAMPLE_STEP)
      if(n_.points[i+j*n_.width].x == n_.points[i+j*n_.width].x  )
        nNormals ++; // if not nan add one valid normal
  
  spx_.reset(new MatrixXf(3,nNormals));
  uint32_t k=0;
  for(uint32_t i=0; i<n_.width; i+=SUBSAMPLE_STEP)
    for(uint32_t j=0; j<n_.height; j+=SUBSAMPLE_STEP)
      if(n_.points[i+j*n_.width].x == n_.points[i+j*n_.width].x )
      {
        (*spx_)(0,k) = n_.points[i+j*n_.width].x;
        (*spx_)(1,k) = n_.points[i+j*n_.width].y;
        (*spx_)(2,k) = n_.points[i+j*n_.width].z;
        assert(fabs(spx_->col(k).norm()-1.0) <1e-3);
        ++k;
      }

  cout<<"# valid Normals: "<< nNormals<<endl;
  cout<<"# valid Normals: "<< spx_->rows()<<" x "<<spx_->cols()<<endl;

  tLog_.tic(1);

  pddpvmf_->nextTimeStep(spx_);
  for(uint32_t i=0; i<nIter_; ++i)
  {
    cout<<" -- iteration = "<<i<<endl;
    pddpvmf_->updateLabelsParallel();
    pddpvmf_->updateCenters();
  }
  pddpvmf_->updateState();
  double residual = 0.0;

  tLog_.toc(1);
  pddpvmf_->getZfromGpu();

  {
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    z_ = pddpvmf_->z();
    K_ = pddpvmf_->getK();
    centroids_ = pddpvmf_->centroids();
    nDisp_ = pcl::PointCloud<pcl::PointXYZRGB>(n_); // copy point cloud
    residual_ = residual;

//    checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w*h*4 *sizeof(float), 
//          cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());
    // update viewer
    update = true;
    updateLock.unlock();
  }
 
  tLog_.toc(2); // total time
  tLog_.logCycle();
  cout<<"---------------------------------------------------------------------------"<<endl;
  tLog_.printStats();
  cout<<" residual="<<residual_<<endl;
  cout<<"---------------------------------------------------------------------------"<<endl;

  fout_<<K_<<" "<<residual_<<endl; fout_.flush();

  return MatrixXf::Zero(3,3);
}


void RealtimeDDPvMF::prepareCUDA(uint32_t w,uint32_t h)
{
  if (cuda_ready) return;
  // CUDA preparations
  printf("Allocating and initializing CUDA arrays...\n");
  checkCudaErrors(cudaMalloc((void **)&d_depth, w * h * sizeof(uint16_t)));
  checkCudaErrors(cudaMalloc((void **)&d_x, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_y, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_z, w * h * sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_n, w * h * X_STEP* sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_xyz, w * h * 4* sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_xu, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_yu, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zu, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_xv, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_yv, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zv, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&a, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&b, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&c, w * h * sizeof(float)));
#ifdef WEIGHTED
  checkCudaErrors(cudaMalloc((void **)&d_weights, w * h * sizeof(float)));
#else
  d_weights = NULL;
#endif
  cout<<"cuda allocations done "<<d_n<<endl;

  h_sobel_dif[0] = 1;
  h_sobel_dif[1] = 0;
  h_sobel_dif[2] = -1;

  h_sobel_sum[0] = 1;
  h_sobel_sum[1] = 2;
  h_sobel_sum[2] = 1;

  // sig =1.0
  // x=np.arange(7) -3.0
  // 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(x*x/sig**2))
  // 0.00443185,  0.05399097,  0.24197072,  0.39894228,  0.24197072,
  // 0.05399097,  0.00443185
  // sig = 2.0
  // 0.0647588 ,  0.12098536,  0.17603266,  0.19947114,  0.17603266,
  // 0.12098536,  0.0647588 
  /*
     h_kernel_avg[0] = 0.00443185;
     h_kernel_avg[1] = 0.05399097;
     h_kernel_avg[2] = 0.24197072;
     h_kernel_avg[3] = 0.39894228;
     h_kernel_avg[4] = 0.24197072;
     h_kernel_avg[5] = 0.05399097;
     h_kernel_avg[6] = 0.00443185;
     */

  h_kernel_avg[0] = 0.0647588;
  h_kernel_avg[1] = 0.12098536;
  h_kernel_avg[2] = 0.17603266;
  h_kernel_avg[3] = 0.19947114;
  h_kernel_avg[4] = 0.17603266;
  h_kernel_avg[5] = 0.12098536;
  h_kernel_avg[6] = 0.0647588;

  n_ = pcl::PointCloud<pcl::PointXYZRGB>(w,h);
  n_cp_ = pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr(&n_);
  Map<MatrixXf, Aligned, OuterStride<> > nMat = 
    n_.getMatrixXfMap(X_STEP,X_STEP,0);
  h_n = nMat.data();//(float *)malloc(w *h *3* sizeof(float));

  pc_ = pcl::PointCloud<pcl::PointXYZ>(w,h);
  pc_cp_ = pcl::PointCloud<pcl::PointXYZ>::ConstPtr(&pc_);
  Map<MatrixXf, Aligned, OuterStride<> > pcMat = 
    pc_.getMatrixXfMap(X_STEP,X_STEP,0);
  h_xyz = pcMat.data();//(float *)malloc(w *h *3* sizeof(float));

  h_dbg = (float *)malloc(w *h * sizeof(float));


  cout<<"inititalizing optSO3"<<endl;
  spx_.reset(new MatrixXf(3,1));
  (*spx_) << 1,0,0; // init just to get the dimensions right.
  if(mode_.compare("dp") == 0)
  {
//#ifndef WEIGHTED
//    optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h);//,d_weights);
//#else
//    optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h,d_weights);
//#endif
//    nCGIter_ = 10; // cannot do that many iterations
//    pddpvmf_ =  new DPvMFMeansCUDA<float>(spx_,lambda_,beta_,Q_,&rndGen_);
  }else if (mode_.compare("ddp") == 0){
    //TODO
//    pddpvmf_ =  new DDPvMFMeans<float>(spx_,lambda_,beta_,Q_,&rndGen_);
    pddpvmf_ =  new DDPvMFMeansCUDA<float>(spx_,lambda_,beta_,Q_,&rndGen_);
//#ifndef WEIGHTED
//    optSO3_ = new OptSO3Approx(25.0f*M_PI/180.0f,d_n,w,h);//,d_weights);
//#else
//    optSO3_ = new OptSO3Approx(25.0f*M_PI/180.0f,d_n,w,h,d_weights);
//#endif
//    nCGIter_ = 25;
  }

  cuda_ready = true;
}

void RealtimeDDPvMF::visualizePc()
{
  // Block signals in this thread
  sigset_t signal_set;
  sigaddset(&signal_set, SIGINT);
  sigaddset(&signal_set, SIGTERM);
  sigaddset(&signal_set, SIGHUP);
  sigaddset(&signal_set, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &signal_set, NULL);

  bool showNormals =true;
  float scale = 2.0f;
  // prepare visualizer named "viewer"
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (
      new pcl::visualization::PCLVisualizer ("3D Viewer"));

  //      viewer->setPointCloudRenderingProperties (
  //          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->initCameraParameters ();
  cv::namedWindow("normals");
  //cv::namedWindow("dbg");
  cv::namedWindow("dbgNan");
  cv::namedWindow("rgb");

//  int v1(0);
//  viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
//  viewer->createViewPort (0.0, 0.0, 1., 1.0, v1);
  viewer->setBackgroundColor (255, 255, 255);
//  viewer->setBackgroundColor (0, 0, 0, v1);
//  viewer->addText ("normals", 10, 10, "v1 text", v1);
  viewer->addCoordinateSystem (1.0);

  viewer->setPosition(0,0);
  viewer->setSize(1000,1000);

//  int v2(0);
//  viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
//  viewer->setBackgroundColor (0.1, 0.1, 0.1, v2);
//  viewer->addText ("pointcloud", 10, 10, "v2 text", v2);
//  viewer->addCoordinateSystem (1.0,v2);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr n;
//  pcl::PointCloud<pcl::PointXYZ>::Ptr pc;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc;

  cv::Mat zIrgb;// (nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
  cv::Mat Icomb;// (nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
  std::stringstream ss;

  Timer t;
  while (!viewer->wasStopped ())
  {
//    cout<<"viewer"<<endl;
    viewer->spinOnce (10);
    cv::waitKey(10);
    // break, if the last update was less than 2s ago
//    if (t.dtFromInit() > 20000.0)
//    {
//      cout<<" ending visualization - waited too long"<<endl;
//      break;
//    }
//    cout<<" after break"<<endl;

    // Get lock on the boolean update and check if cloud was updated
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    if (updateRGB_)
    {
      cout<<"show rgb"<<endl;
      imshow("rgb",rgb_);
#ifdef DUMP_FRAMES
      ss.str(""); ss<<resultsPath_<<"rgb_"<< setw(5) << setfill('0') <<nFrame_<<".jpg";
      cout<<"writing "<< resultsPath_<< "    "<<ss.str()<<endl;
      cv::imwrite(ss.str(),rgb_);
#endif
      updateRGB_ = false;
      t=Timer();
    } 
    if (update)
    {

//      cout<<"show pc"<<endl;
//      ss.str("residual=");
//      ss <<residual_;
//      if(!viewer->updateText(ss.str(),10,20,"residual"))
//        viewer->addText(ss.str(),10,20, "residual", v1);


      cv::Mat nI(n_.height,n_.width,CV_32FC3); 
      for(uint32_t i=0; i<n_.width; ++i)
        for(uint32_t j=0; j<n_.height; ++j)
        {
          // nI is BGR but I want R=x G=y and B=z
          nI.at<cv::Vec3f>(j,i)[0] = (1.0f+n_.points[i+j*n_.width].z)*0.5f; // to match pc
          nI.at<cv::Vec3f>(j,i)[1] = (1.0f+n_.points[i+j*n_.width].y)*0.5f; 
          nI.at<cv::Vec3f>(j,i)[2] = (1.0f+n_.points[i+j*n_.width].x)*0.5f; 
        }
      cv::imshow("normals",nI); 

      uint32_t Kmax = 4;
      uint32_t k=0;
//      cv::Mat Iz(nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC1); 
      zIrgb = cv::Mat(nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
      for(uint32_t i=0; i<nDisp_.width; i+=SUBSAMPLE_STEP)
        for(uint32_t j=0; j<nDisp_.height; j+=SUBSAMPLE_STEP)
          if(nDisp_.points[i+j*nDisp_.width].x == nDisp_.points[i+j*nDisp_.width].x )
          {
            uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[0] = JET_b[idz]*255;    
            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[1] = JET_g[idz]*255;    
            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[2] = JET_r[idz]*255;    
            k++;
          }else{
            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[0] = 255;
            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[1] = 255;    
            zIrgb.at<cv::Vec3b>(j/SUBSAMPLE_STEP,i/SUBSAMPLE_STEP)[2] = 255;    
          }

      cv::addWeighted(rgb_ , 0.7, zIrgb, 0.3, 0.0, Icomb);
      cv::imshow("dbg",Icomb); 

      if(showNormals){
        n = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>(k,1));
        uint32_t nPoints = k;
        k=0;
        for(uint32_t i=0; i<nDisp_.width; i+=SUBSAMPLE_STEP)
          for(uint32_t j=0; j<nDisp_.height; j+=SUBSAMPLE_STEP)
            if(nDisp_.points[i+j*nDisp_.width].x == nDisp_.points[i+j*nDisp_.width].x )
            {
              uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
              n->points[k] = nDisp_.points[i+j*nDisp_.width];
              n->points[k].r = JET_r[idz]*255;
              n->points[k].g = JET_g[idz]*255;
              n->points[k].b = JET_b[idz]*255;
//              n->push_back(pcl::PointXYZL());
//              nDisp_.points[i].x,nDisp_.points[i].y,nDisp_.points[i].z,z_(k)));
              k++;
            };
        //pcl::transformPointCloud(*n, *n, wTk);
        
        pc = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>(K_,1));
        for(uint32_t k=0; k<K_; ++k)
        {
          pc->points[k].x = centroids_(0,k);
        }

      }

//      pc = pcl::PointCloud<pcl::PointXYZ>::Ptr(
//          new pcl::PointCloud<pcl::PointXYZ>);
//      for(uint32_t i=0; i<pc_.width; i+= 5)
//        for(uint32_t j=0; j<pc_.height; j+=5)
//          pc->points.push_back(pc_cp_->points[i+j*pc_.width]);
//      pcl::transformPointCloud(*pc, *pc , wTk);

//      if(!updateCosy(viewer, kRw_,"mf",2.0f))
//        addCosy(viewer,kRw_,"mf",2.0f, v1);
//
      if(showNormals)
        if(!viewer->updatePointCloud(n, "normals"))
          viewer->addPointCloud(n, "normals");
//
//      if(!viewer->updatePointCloud(pc, "pc"))
//        viewer->addPointCloud(pc, "pc",v2);


//        viewer->setBackgroundColor (255, 255, 255);
#ifdef DUMP_FRAMES
      ss.str(""); ss<<resultsPath_<<"2Dsegmentation_"<< setw(5) << setfill('0') <<nFrame_<<".png";
      cout<<"writing "<<ss.str()<<endl;
      cv::imwrite(ss.str(),zIrgb);

      ss.str(""); ss<<resultsPath_<<"pcNormals_"<< setw(5) << setfill('0') <<nFrame_<<".png";
      cout<<"writing "<<ss.str()<<endl;
      viewer->saveScreenshot(ss.str());
#endif

      // Screenshot  
//      vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = 
//        vtkSmartPointer<vtkWindowToImageFilter>::New();
//      windowToImageFilter->SetInput(dynamic_cast<vtkWindow*>(viewer->getRenderWindow().GetPointer()));
//      windowToImageFilter->SetMagnification(3); //set the resolution of the output image (3 times the current resolution of vtk render window)
//      windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
//      windowToImageFilter->Update();
//
//      vtkSmartPointer<vtkPNGWriter> writer = 
//        vtkSmartPointer<vtkPNGWriter>::New();
//      writer->SetFileName("screenshot2.png");
//      writer->SetInputConnection(windowToImageFilter->GetOutputPort());
//      writer->Write();
 

      nFrame_ ++;
      update = false;
      t=Timer();
    }
    updateLock.unlock();
  }
}

void RealtimeDDPvMF::getAxisAssignments()
{
  
}

void RealtimeDDPvMF::computeDerivatives(uint32_t w,uint32_t h)
{
  setConvolutionKernel_small(h_sobel_dif);
  convolutionRowsGPU_small(a,d_x,w,h);
  convolutionRowsGPU_small(b,d_y,w,h);
  convolutionRowsGPU_small(c,d_z,w,h);
  setConvolutionKernel_small(h_sobel_sum);
  convolutionColumnsGPU_small(d_xu,a,w,h);
  convolutionColumnsGPU_small(d_yu,b,w,h);
  convolutionColumnsGPU_small(d_zu,c,w,h);
  convolutionRowsGPU_small(a,d_x,w,h);
  convolutionRowsGPU_small(b,d_y,w,h);
  convolutionRowsGPU_small(c,d_z,w,h);
  setConvolutionKernel_small(h_sobel_dif);
  convolutionColumnsGPU_small(d_xv,a,w,h);
  convolutionColumnsGPU_small(d_yv,b,w,h);
  convolutionColumnsGPU_small(d_zv,c,w,h);
}

void RealtimeDDPvMF::smoothDerivatives(uint32_t iterations, uint32_t w,uint32_t h)
{
  setConvolutionKernel(h_kernel_avg);
  for(uint32_t i=0; i<iterations; ++i)
  {
    convolutionRowsGPU(a,d_xu,w,h);
    convolutionRowsGPU(b,d_yu,w,h);
    convolutionRowsGPU(c,d_zu,w,h);
    convolutionColumnsGPU(d_xu,a,w,h);
    convolutionColumnsGPU(d_yu,b,w,h);
    convolutionColumnsGPU(d_zu,c,w,h);
    convolutionRowsGPU(a,d_xv,w,h);
    convolutionRowsGPU(b,d_yv,w,h);
    convolutionRowsGPU(c,d_zv,w,h);
    convolutionColumnsGPU(d_xv,a,w,h);
    convolutionColumnsGPU(d_yv,b,w,h);
    convolutionColumnsGPU(d_zv,c,w,h);
  }
}

//#define BILATERAL
void RealtimeDDPvMF::depth2smoothXYZ(float invF, uint32_t w,uint32_t h)
{
  cout<<"with "<<w<<" "<<h<<endl;
  depth2floatGPU(d_depth,a,w,h);

//  for(uint32_t i=0; i<3; ++i)
//  {
//    depthFilterGPU(a,w,h);
//  }

  //TODO compare:
  // now smooth the derivatives
#ifdef BILATERAL
  cout<<"bilateral with "<<w<<" "<<h<<endl;
  bilateralFilterGPU(a,b,w,h,6,20.0,0.05);
  // convert depth into x,y,z coordinates
  depth2xyzFloatGPU(b,d_x,d_y,d_z,invF,w,h,d_xyz); 
#else
  setConvolutionKernel(h_kernel_avg);
  for(uint32_t i=0; i<3; ++i)
  {
    convolutionRowsGPU(b,a,w,h);
    convolutionColumnsGPU(a,b,w,h);
  }
  // convert depth into x,y,z coordinates
  depth2xyzFloatGPU(a,d_x,d_y,d_z,invF,w,h,d_xyz); 
#endif


}


void RealtimeDDPvMF::run ()
{
  boost::thread visualizationThread(&RealtimeDDPvMF::visualizePc,this); 

  this->run_impl();
  while (42) boost::this_thread::sleep (boost::posix_time::seconds (1));
  this->run_cleanup_impl();
  visualizationThread.join();
}

