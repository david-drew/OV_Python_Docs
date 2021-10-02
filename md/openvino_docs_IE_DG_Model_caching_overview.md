# Model Caching Overview

## Introduction

As described in Inference Engine Developer Guide, common application flow consists of the following steps:

1. Create an Inference Engine Core object
2. *Read the Intermediate Representation* - Read an Intermediate Representation file into an object of the [ie_api.IENetwork](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html)
3. *Prepare inputs and outputs*
4. *Set configuration* - Pass device-specific loading configurations to the device
5. *Compile and Load Network to device* - Use the IECore.load_network() method and specify the target device
6. *Set input data*
7. *Execute the model* - Run inference

Step #5 can potentially perform several time-consuming device-specific optimizations and network compilations, and such delays can lead to bad user experience on application startup. To avoid this, some devices offer Import/Export network capability, and it is possible to either use the [Compile Tool](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_compile_tool_README.html) or enable model caching to export compiled network automatically. Reusing cached networks can significantly reduce load network time.





