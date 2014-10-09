#include <realtimeDDPvMF.hpp>

RealtimeDDPvMF::RealtimeDDPvMF(std::string mode) 
  :  tLog_("./timer.log",3,"TimerLog"),
  residual_(0.0), nIter_(3), 
  nFrame_(0), resultsPath_("../results/"),
  normalExtractor_(570.f),
  fout_("./stats.log",ofstream::out),
  update(false), updateRGB_(false), 
  mode_(mode), 
  lambda_(cos(93.0*M_PI/180.0)-1.), beta_(3.e5), Q_(-1.e3),
  rndGen_(91),
  JET_r ({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00588235294117645,0.02156862745098032,0.03725490196078418,0.05294117647058827,0.06862745098039214,0.084313725490196,0.1000000000000001,0.115686274509804,0.1313725490196078,0.1470588235294117,0.1627450980392156,0.1784313725490196,0.1941176470588235,0.2098039215686274,0.2254901960784315,0.2411764705882353,0.2568627450980392,0.2725490196078431,0.2882352941176469,0.303921568627451,0.3196078431372549,0.3352941176470587,0.3509803921568628,0.3666666666666667,0.3823529411764706,0.3980392156862744,0.4137254901960783,0.4294117647058824,0.4450980392156862,0.4607843137254901,0.4764705882352942,0.4921568627450981,0.5078431372549019,0.5235294117647058,0.5392156862745097,0.5549019607843135,0.5705882352941174,0.5862745098039217,0.6019607843137256,0.6176470588235294,0.6333333333333333,0.6490196078431372,0.664705882352941,0.6803921568627449,0.6960784313725492,0.7117647058823531,0.7274509803921569,0.7431372549019608,0.7588235294117647,0.7745098039215685,0.7901960784313724,0.8058823529411763,0.8215686274509801,0.8372549019607844,0.8529411764705883,0.8686274509803922,0.884313725490196,0.8999999999999999,0.9156862745098038,0.9313725490196076,0.947058823529412,0.9627450980392158,0.9784313725490197,0.9941176470588236,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9862745098039216,0.9705882352941178,0.9549019607843139,0.93921568627451,0.9235294117647062,0.9078431372549018,0.892156862745098,0.8764705882352941,0.8607843137254902,0.8450980392156864,0.8294117647058825,0.8137254901960786,0.7980392156862743,0.7823529411764705,0.7666666666666666,0.7509803921568627,0.7352941176470589,0.719607843137255,0.7039215686274511,0.6882352941176473,0.6725490196078434,0.6568627450980391,0.6411764705882352,0.6254901960784314,0.6098039215686275,0.5941176470588236,0.5784313725490198,0.5627450980392159,0.5470588235294116,0.5313725490196077,0.5156862745098039,0.5}),
  JET_g ({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001960784313725483,0.01764705882352935,0.03333333333333333,0.0490196078431373,0.06470588235294117,0.08039215686274503,0.09607843137254901,0.111764705882353,0.1274509803921569,0.1431372549019607,0.1588235294117647,0.1745098039215687,0.1901960784313725,0.2058823529411764,0.2215686274509804,0.2372549019607844,0.2529411764705882,0.2686274509803921,0.2843137254901961,0.3,0.3156862745098039,0.3313725490196078,0.3470588235294118,0.3627450980392157,0.3784313725490196,0.3941176470588235,0.4098039215686274,0.4254901960784314,0.4411764705882353,0.4568627450980391,0.4725490196078431,0.4882352941176471,0.503921568627451,0.5196078431372548,0.5352941176470587,0.5509803921568628,0.5666666666666667,0.5823529411764705,0.5980392156862746,0.6137254901960785,0.6294117647058823,0.6450980392156862,0.6607843137254901,0.6764705882352942,0.692156862745098,0.7078431372549019,0.723529411764706,0.7392156862745098,0.7549019607843137,0.7705882352941176,0.7862745098039214,0.8019607843137255,0.8176470588235294,0.8333333333333333,0.8490196078431373,0.8647058823529412,0.8803921568627451,0.8960784313725489,0.9117647058823528,0.9274509803921569,0.9431372549019608,0.9588235294117646,0.9745098039215687,0.9901960784313726,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9901960784313726,0.9745098039215687,0.9588235294117649,0.943137254901961,0.9274509803921571,0.9117647058823528,0.8960784313725489,0.8803921568627451,0.8647058823529412,0.8490196078431373,0.8333333333333335,0.8176470588235296,0.8019607843137253,0.7862745098039214,0.7705882352941176,0.7549019607843137,0.7392156862745098,0.723529411764706,0.7078431372549021,0.6921568627450982,0.6764705882352944,0.6607843137254901,0.6450980392156862,0.6294117647058823,0.6137254901960785,0.5980392156862746,0.5823529411764707,0.5666666666666669,0.5509803921568626,0.5352941176470587,0.5196078431372548,0.503921568627451,0.4882352941176471,0.4725490196078432,0.4568627450980394,0.4411764705882355,0.4254901960784316,0.4098039215686273,0.3941176470588235,0.3784313725490196,0.3627450980392157,0.3470588235294119,0.331372549019608,0.3156862745098041,0.2999999999999998,0.284313725490196,0.2686274509803921,0.2529411764705882,0.2372549019607844,0.2215686274509805,0.2058823529411766,0.1901960784313728,0.1745098039215689,0.1588235294117646,0.1431372549019607,0.1274509803921569,0.111764705882353,0.09607843137254912,0.08039215686274526,0.06470588235294139,0.04901960784313708,0.03333333333333321,0.01764705882352935,0.001960784313725483,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}),
  JET_b ( {0.5,0.5156862745098039,0.5313725490196078,0.5470588235294118,0.5627450980392157,0.5784313725490196,0.5941176470588235,0.6098039215686275,0.6254901960784314,0.6411764705882352,0.6568627450980392,0.6725490196078432,0.6882352941176471,0.7039215686274509,0.7196078431372549,0.7352941176470589,0.7509803921568627,0.7666666666666666,0.7823529411764706,0.7980392156862746,0.8137254901960784,0.8294117647058823,0.8450980392156863,0.8607843137254902,0.8764705882352941,0.892156862745098,0.907843137254902,0.9235294117647059,0.9392156862745098,0.9549019607843137,0.9705882352941176,0.9862745098039216,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9941176470588236,0.9784313725490197,0.9627450980392158,0.9470588235294117,0.9313725490196079,0.915686274509804,0.8999999999999999,0.884313725490196,0.8686274509803922,0.8529411764705883,0.8372549019607844,0.8215686274509804,0.8058823529411765,0.7901960784313726,0.7745098039215685,0.7588235294117647,0.7431372549019608,0.7274509803921569,0.7117647058823531,0.696078431372549,0.6803921568627451,0.6647058823529413,0.6490196078431372,0.6333333333333333,0.6176470588235294,0.6019607843137256,0.5862745098039217,0.5705882352941176,0.5549019607843138,0.5392156862745099,0.5235294117647058,0.5078431372549019,0.4921568627450981,0.4764705882352942,0.4607843137254903,0.4450980392156865,0.4294117647058826,0.4137254901960783,0.3980392156862744,0.3823529411764706,0.3666666666666667,0.3509803921568628,0.335294117647059,0.3196078431372551,0.3039215686274508,0.2882352941176469,0.2725490196078431,0.2568627450980392,0.2411764705882353,0.2254901960784315,0.2098039215686276,0.1941176470588237,0.1784313725490199,0.1627450980392156,0.1470588235294117,0.1313725490196078,0.115686274509804,0.1000000000000001,0.08431372549019622,0.06862745098039236,0.05294117647058805,0.03725490196078418,0.02156862745098032,0.00588235294117645,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0})
{

  cout<<"inititalizing optSO3"<<endl;
  spx_.reset(new MatrixXf(3,1));
  (*spx_) << 1,0,0; // init just to get the dimensions right.
  if(mode_.compare("dp") == 0)
  {
    //    pddpvmf_ =  new DPvMFMeansCUDA<float>(spx_,lambda_,beta_,Q_,&rndGen_);
  }else if (mode_.compare("ddp") == 0){
    //TODO
    //    pddpvmf_ =  new DDPvMFMeans<float>(spx_,lambda_,beta_,Q_,&rndGen_);
    pddpvmf_ =  new DDPvMFMeansCUDA<float>(spx_,lambda_,beta_,Q_,&rndGen_);
  }
}

RealtimeDDPvMF::~RealtimeDDPvMF()
{
  delete pddpvmf_;
  fout_.close();
}

#define SMOOTH_DEPTH

Matrix3f RealtimeDDPvMF::depth_cb(const uint16_t *data, int w, int h) 
{
  Timer t0;
  tLog_.tic(-1); // reset all timers
  normalExtractor_.compute(data,w,h);
  tLog_.toc(0); 
  n_cp_ = normalExtractor_.normals();

#ifdef RM_NANS_FROM_DEPTH
  uint32_t nNormals = 0;
  for(uint32_t i=0; i<n_cp_->width; i+=SUBSAMPLE_STEP)
    for(uint32_t j=0; j<n_cp_->height; j+=SUBSAMPLE_STEP)
      if(n_cp_->points[i+j*n_cp_->width].x == n_cp_->points[i+j*n_cp_->width].x  )
        nNormals ++; // if not nan add one valid normal
  
  spx_.reset(new MatrixXf(3,nNormals));
  uint32_t k=0;
  for(uint32_t i=0; i<n_cp_->width; i+=SUBSAMPLE_STEP)
    for(uint32_t j=0; j<n_cp_->height; j+=SUBSAMPLE_STEP)
      if(n_cp_->points[i+j*n_cp_->width].x == n_cp_->points[i+j*n_cp_->width].x )
      {
        (*spx_)(0,k) = n_cp_->points[i+j*n_cp_->width].x;
        (*spx_)(1,k) = n_cp_->points[i+j*n_cp_->width].y;
        (*spx_)(2,k) = n_cp_->points[i+j*n_cp_->width].z;
        assert(fabs(spx_->col(k).norm()-1.0) <1e-3);
        ++k;
      }

  cout<<"# valid Normals: "<< nNormals<<endl;
  cout<<"# valid Normals: "<< spx_->rows()<<" x "<<spx_->cols()<<endl;
  pddpvmf_->nextTimeStep(spx_);

#else
//  uint32_t nNormals = 0;
//  for(uint32_t i=0; i<n_cp_->width; i+=SUBSAMPLE_STEP)
//    for(uint32_t j=0; j<n_cp_->height; j+=SUBSAMPLE_STEP)
//      if(n_cp_->points[i+j*n_cp_->width].x != n_cp_->points[i+j*n_cp_->width].x  )
//        nNormals ++; // if not nan add one valid normal
//  cout<<"#nans = "<<nNormals<<endl;
  pddpvmf_->nextTimeStep(normalExtractor_.d_normals(),w*h,
      normalExtractor_.d_step(), normalExtractor_.d_offset());
#endif

  tLog_.tic(1);
  cout<<"iterating now"<<endl;
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
    nDisp_ = pcl::PointCloud<pcl::PointXYZRGB>(*n_cp_); // copy point cloud
    residual_ = residual;

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

  return MatrixXf::Identity(3,3);
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

//  viewer->setPosition(0,0);
//  viewer->setSize(1000,1000);

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


      cv::Mat nI(n_cp_->height,n_cp_->width,CV_32FC3); 
      for(uint32_t i=0; i<n_cp_->width; ++i)
        for(uint32_t j=0; j<n_cp_->height; ++j)
        {
          // nI is BGR but I want R=x G=y and B=z
          nI.at<cv::Vec3f>(j,i)[0] = (1.0f+n_cp_->points[i+j*n_cp_->width].z)*0.5f; // to match pc
          nI.at<cv::Vec3f>(j,i)[1] = (1.0f+n_cp_->points[i+j*n_cp_->width].y)*0.5f; 
          nI.at<cv::Vec3f>(j,i)[2] = (1.0f+n_cp_->points[i+j*n_cp_->width].x)*0.5f; 
        }
      cv::imshow("normals",nI); 

      uint32_t Kmax = 4;
      uint32_t k=0;
      cout<<" z shape "<<z_.rows()<<" "<< nDisp_.width<<" " <<nDisp_.height<<endl;
//      cv::Mat Iz(nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC1); 
      zIrgb = cv::Mat(nDisp_.height/SUBSAMPLE_STEP,nDisp_.width/SUBSAMPLE_STEP,CV_8UC3);
      for(uint32_t i=0; i<nDisp_.width; i+=SUBSAMPLE_STEP)
        for(uint32_t j=0; j<nDisp_.height; j+=SUBSAMPLE_STEP)
          if(nDisp_.points[i+j*nDisp_.width].x == nDisp_.points[i+j*nDisp_.width].x )
          {
#ifdef RM_NANS_FROM_DEPTH
            uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
#else
            uint8_t idz = (static_cast<uint8_t>(z_(nDisp_.width*j +i)))*255/Kmax;
#endif
//            cout<<"k "<<k<<" "<< z_.rows() <<"\t"<<z_(k)<<"\t"<<int32_t(idz)<<endl;
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
//              k = nDisp_.width*j +i;
#ifdef RM_NANS_FROM_DEPTH
            uint8_t idz = (static_cast<uint8_t>(z_(k)))*255/Kmax;
#else
            uint8_t idz = (static_cast<uint8_t>(z_(nDisp_.width*j +i)))*255/Kmax;
#endif
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

void RealtimeDDPvMF::run ()
{
  boost::thread visualizationThread(&RealtimeDDPvMF::visualizePc,this); 

  this->run_impl();
  while (42) boost::this_thread::sleep (boost::posix_time::seconds (1));
  this->run_cleanup_impl();
  visualizationThread.join();
}

