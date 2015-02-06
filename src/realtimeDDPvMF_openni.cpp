
#include <iostream>
#include <string>
//#include <realtimeDDPvMF_openni.hpp>
#include <rtDDPvMF.hpp>

// Utilities and system includes
//#include <helper_functions.h>
#include <nvidia/helper_cuda.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main (int argc, char** argv)
{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("K,K", po::value<int>(), "K for spkm clustering")
    ("mode", po::value<string>(), "mode of the rtDDPvMF (spkm, dp, ddp)")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  string mode = "ddp";
  if(vm.count("mode")) mode = vm["mode"].as<string>();

  findCudaDevice(argc,(const char**)argv);

  if(mode.compare("spkm") == 0)
  {
    RealtimeSpkm v(540.,0.2*0.2,10,);
    v.run ();
  }else{
    RealtimeDDPvMF v(mode,540.,0.2*0.2,10);
    v.run ();
  }
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
