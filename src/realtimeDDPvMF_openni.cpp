
#include <iostream>
#include <string>
#include <realtimeDDPvMF_openni.hpp>

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
    ("mode", po::value<string>(), 
    "mode of the rtDDPvMF (direct, approx)")
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

  RealtimeDDPvMF_openni v(mode);
  v.run ();
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
