/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <string>
#include <rtDDPvMF/rtDDPvMF.hpp>
#include <rtDDPvMF/rtSpkm.hpp>
#include <rtDDPvMF/realtimeDDPvMF_openni.hpp>
#include <rtDDPvMF/rtSpkm_openni.hpp>

// Utilities and system includes
//#include <helper_functions.h>
//#include <nvidia/helper_cuda.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main (int argc, char** argv)
{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("K,K", po::value<int>(), "K for spkm clustering")
    ("lambdaDeg,l", po::value<double>(), "lambda in degree for dp and ddp")
    ("beta,b", po::value<double>(), "beta parameter of the ddp")
    ("nFramesSurvive,s", po::value<int>(), "number of frames a cluster survives without observation")
//    ("mode", po::value<string>(), "mode of the rtDDPvMF (spkm, dp, ddp)")
    ("in,i", po::value<string>(), "path to input file")
    ("out,o", po::value<string>(), "path to output file")
    ("display,d", "display results")
    ("B,B", po::value<int>(), "B for guided filter")
    ("eps", po::value<float>(), "eps for guided filter")
    ("f_d,f", po::value<float>(), "focal length of depth camera")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  CfgRtDDPvMF cfg;
  cfg.f_d = 540.;
  cfg.beta = 1e5;
  cfg.nFramesSurvive_ = 300;
//  cfg.nSkipFramesSave = 60;
  cfg.pathOut = std::string("../results/");
  double lambdaDeg = 93.;
  int K = -1;
  if(vm.count("K")) K = vm["K"].as<int>();
  if(vm.count("lambdaDeg")) lambdaDeg = vm["lambdaDeg"].as<double>();
  if(vm.count("nFramesSurvive")) cfg.nFramesSurvive_ = vm["nFramesSurvive"].as<int>();
  if(vm.count("beta")) cfg.beta = vm["beta"].as<double>();

  cfg.lambdaFromDeg(lambdaDeg);
  cfg.QfromFrames2Survive(cfg.nFramesSurvive_);

  if(cfg.nFramesSurvive_==0)
    cfg.pathOut += "dp_fbf/";
  else if(cfg.nFramesSurvive_==1)
    cfg.pathOut += "dp/";
  else if(K>0)
    cfg.pathOut += "spkm/";
  else
    cfg.pathOut += "ddp/";

  string path = "";
//  string mode = "";
  cudaPcl::CfgSmoothNormals cfgNormals;
  cfgNormals.f_d = 540.;
  cfgNormals.eps = 0.2*0.2;
  cfgNormals.B = 9;
  cfgNormals.compress = true;
  uint32_t T = 10;
//  if(vm.count("mode")) mode = vm["mode"].as<string>();
  if(vm.count("in")) path = vm["in"].as<string>();
  if(vm.count("eps")) cfgNormals.eps = vm["eps"].as<float>();
  if(vm.count("f_d")) cfgNormals.f_d = vm["f_d"].as<float>();
  if(vm.count("B")) cfgNormals.B = uint32_t( vm["B"].as<int>());

  findCudaDevice(argc,(const char**)argv);
  shared_ptr<RtDDPvMF> pRtDDPvMF;
  shared_ptr<RtSpkm> pRtSpkm;

  if(K<0)
  {
    pRtDDPvMF = shared_ptr<RtDDPvMF>(new RtDDPvMF(cfg,cfgNormals));
  }else{
    pRtSpkm = shared_ptr<RtSpkm>(new RtSpkm(cfg.pathOut,K,cfgNormals));
  }
  if (path.length() == 0)
  {
    cout<<"no input path -> trying openni"<<endl;
    if(K<0)
    {
      cout<<"rtDDPvMFmeans lambdaDeg="<<cfg.lambdaDeg_<<" beta="<<cfg.beta
        <<"nFramesSurvive="<<cfg.nFramesSurvive_<<endl;
      cout<<"output path: "<<cfg.pathOut<<endl;
      RealtimeDDPvMF v(pRtDDPvMF);
      v.run ();
    }else{
      cout<<"rtSpkm K="<<K<<endl;
      cout<<"output path: "<<cfg.pathOut<<endl;
      RealtimeSpkm v(pRtSpkm);
      v.run ();
    }
  }else{
    cout<<"reading depth image from "<<path<<endl;
    cv::Mat depth = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
    cout<<"type: "<<int(depth.type()) <<" != "<<int(CV_16U) <<endl;

    string pathRgb(path);
    pathRgb.replace(path.length()-5,1,"rgb");
    cout<<"reading rgb image from "<<pathRgb<<endl;
    cv::Mat gray = cv::imread(pathRgb, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat rgb = cv::imread(pathRgb);

    if(vm.count("display")) 
    {
      cv::Mat dI(depth.rows,depth.cols,CV_8UC1);
      depth.convertTo(dI,CV_8UC1,255./4000.,-19.);
      cv::imshow("d",dI);
      cv::imshow("rgb",rgb);
      cv::waitKey(0);
    }
    cv::Mat dI;
    cv::Mat nI;
    cv::Mat zI;
    cv::Mat Iout;
    MatrixXf centroids;

    if(K<0)
    {
      cout<<"rtDDPvMFmeans lambdaDeg="<<cfg.lambdaDeg_<<" beta="<<cfg.beta
        <<"nFramesSurvive="<<cfg.nFramesSurvive_<<endl;
      cout<<"output path: "<<cfg.pathOut<<endl;
      for(uint32_t i=0; i<T; ++i)
        pRtDDPvMF->compute(reinterpret_cast<uint16_t*>(depth.data),
            depth.cols,depth.rows);
      Iout = pRtDDPvMF->overlaySeg(rgb);
//      cv::Mat Iout = pRtDDPvMF->overlaySeg(gray);
      if(vm.count("display")) 
      {
        dI = pRtDDPvMF->smoothDepthImg();
        nI = pRtDDPvMF->normalsImg();
        zI = pRtDDPvMF->labelsImg();
      }
      centroids = pRtDDPvMF->centroids();

    }else{
      cout<<"rtSpkm K="<<K<<endl;
      cout<<"output path: "<<cfg.pathOut<<endl;
      for(uint32_t i=0; i<T; ++i)
        pRtSpkm->compute(reinterpret_cast<uint16_t*>(depth.data),
            depth.cols,depth.rows);
      Iout = pRtSpkm->overlaySeg(rgb);
//      cv::Mat Iout = pRtSpkm->overlaySeg(gray);
      if(vm.count("display")) 
      {
        dI = pRtSpkm->smoothDepthImg();
        nI = pRtSpkm->normalsImg();
        zI = pRtSpkm->labelsImg();
      }
      centroids = pRtSpkm->centroids();
    }

    if(vm.count("display")) 
    {
      cv::imshow("dS",dI);
      cv::imshow("normals",nI);
      cv::imshow("zI",zI);
      cv::imshow("out",Iout);
      cv::waitKey(0);
    }

    if(vm.count("out"))
    {
      cout<<" writing out put to "<<endl
        <<(vm["out"].as<string>()+"_rgbLabels.png")<<endl
        <<vm["out"].as<string>()+"_cRmf.csv"<<endl;
      cv::imwrite(vm["out"].as<string>()+"_rgbLabels.png",Iout);
      ofstream out((vm["out"].as<string>()+"_cRmf.csv").data(),
          ofstream::out);
      for(uint32_t i=0; i<centroids.rows();++i) 
      {
        for(uint32_t j=0; j<centroids.cols()-1;++j) 
          out << centroids(i,j)<<" ";
        out << centroids(i,centroids.cols()-1)<<endl;
      }
      out.close();
    }
  }
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
