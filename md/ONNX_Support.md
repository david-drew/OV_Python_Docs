# ONNX Format Support in the OpenVINO™ Toolkit
  
Starting with the 2020.4 release, OpenVINO™ supports reading native ONNX models in addition to [converting ONNX models](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html) with the Model Optimizer. The `IECore.read_network()` method provides a uniform way to read models from IR or ONNX format, it is a recommended approach to reading models. Example:
  
```
  from openvino.inference_engine import IECore, StatusCode
  ie = IECore();
  net = ie.read_network(model="model.onnx");
```

## Reshape Feature
OpenVINO™ doesn’t provide a mechanism to specify pre-processing (like mean values subtraction, reverse input channels) for the ONNX format. If an ONNX model contains dynamic shapes for input, please use the [IENetwork.reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a) method for shape specialization.

## Weights Saved in External Files

OpenVINO™ supports ONNX models that store weights in external files. It is especially useful for models larger than 2GB because of protobuf limitations. To read such models, use the `model` parameter in the `IECore.read_network(model='path_to_model.onnx')` method. Note that the parameter for the path to the binary weight file, `weights=` should be empty in this case, because paths to external weights are saved directly in an ONNX model. Otherwise, a runtime exception is thrown. Reading models with external weights is **NOT** supported by the `read_network(weights="file.bin")` parameter.

Paths to external weight files are saved in an ONNX model; these paths are relative to the model’s directory path. It means that if a model is located at: home/user/workspace/models/model.onnx and a file that contains external weights: /home/user/workspace/models/data/weights.bin the path saved in model should be: data/weights.bin.

> **NOTE**
* A single model can use many external weights files.
* Data of many tensors can be stored in a single external weights file (it is processed using offset and length values, which can be also saved in a model).

The described mechanism is the only possibility to read weights from external files. The following input parameter of the [IECore.read_network](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a0d69c298618fab3a08b855442dca430f) function overloads is NOT supported for ONNX models and should be passed empty:
* weights

### Unsupported types of tensors
* string
* complex64
* complex128
