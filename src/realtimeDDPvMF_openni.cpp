
#include <iostream>
#include <string>
//#include <realtimeDDPvMF_openni.hpp>
#include <rtDDPvMF.hpp>
#include <rtSpkm.hpp>

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
    ("mode", po::value<string>(), "mode of the rtDDPvMF (spkm, dp, ddp)")
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

  findCudaDevice(argc,(const char**)argv);
  if(K<0)
  {
    cout<<"rtDDPvMFmeans lambdaDeg="<<cfg.lambdaDeg_<<" beta="<<cfg.beta
      <<"nFramesSurvive="<<cfg.nFramesSurvive_<<endl;
    RealtimeDDPvMF v(cfg,0.2*0.2,10);
    v.run ();
  }else{
    cout<<"rtSpkm K="<<K<<endl;
    cout<<"output path: "<<cfg.pathOut<<endl;
    RealtimeSpkm v(cfg.pathOut,540.,0.2*0.2,10,K);
    v.run ();
  }
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
