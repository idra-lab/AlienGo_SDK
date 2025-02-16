# Aliengo SDK with Neural Network Control

This repository is an extension of the original SDK repository for Unitree AlienGo robot. It contains the original components, such as example files both in python and C++. The extension is in the implementation of external nn for the robot control on low level commands.

## Supported Robot
- Aliengo

### Dependencies
* [Unitree](https://www.unitree.com/download)
```bash
Legged_sport    >= v1.36.0
firmware H0.1.7 >= v0.1.35
         H0.1.9 >= v0.1.35
```
* [Boost](http://www.boost.org) (version 1.5.4 or higher)
* [CMake](http://www.cmake.org) (version 2.8.3 or higher)
* [g++](https://gcc.gnu.org/) (version 8.3.0 or higher)

### Build
```bash
mkdir build
cd build
cmake ..
make
```

If you want to build the python wrapper, then replace the cmake line with:
```bash
cmake -DPYTHON_BUILD=TRUE ..
```

If can not find pybind11 headers, then add
```bash
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pybind11/include)
```
at line 14 in python_wrapper/CMakeLists.txt.

If can not find msgpack.hpp, then
```bash
sudo apt install libmsgpack*
```

## Setting Up Environment Variables

Add the following lines to your shell configuration file (e.g., `~/.bashrc`):

```bash
# Set the SDK path (replace YOUR_SDK_PATH with the actual path)
export LD_LIBRARY_PATH="$HOME/YOUR_SDK_PATH/lib/python/amd64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PYTHONPATH:/YOUR_SDK_PATH/lib/python/amd64/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/YOUR_SDK_PATH/lib/cpp/amd64/"
```

## Run

#### C++  
Run examples with `sudo` to enable memory locking.  

#### Python  
##### ARM  
Modify the following line:  
`sys.path.append('../lib/python/amd64')` → `sys.path.append('../lib/python/arm64')`  

The main script to run is **`position_nn.py`**, located in the `example_py` folder.  


# General infos

## Neural Networks

Various control policies are stored in the `nn` folder.  
To switch between different policies, update the `checkpoint_path` field in the `config.yaml` file.  
This determines which policy is used for control.  

## Leg Order Mapping

When using the Unitree Legged SDK, the order of the legs differs from the expected input order of the neural network. To ensure compatibility, the legs are reordered as follows:

- **SDK Order:** `[FR, FL, RR, RL]` (Front Right, Front Left, Rear Right, Rear Left)  
- **Neural Network Order:** `[FL, FR, RL, RR]` (Front Left, Front Right, Rear Left, Rear Right)  

The function **`swap_legs()`** is responsible for performing this reordering, ensuring that the data is correctly mapped before being processed by the neural network.

