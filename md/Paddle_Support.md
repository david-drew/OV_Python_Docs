# Paddle Support in the OpenVINO™ {#openvino_docs_IE_DG_Paddle_Support}

Starting from the 2022.1 release, OpenVINO™ supports reading native Paddle models.
[IECore.read_network](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a0d69c298618fab3a08b855442dca430f) method provides a uniform way to read models from IR or Paddle format, it is a recommended approach to reading models.

## Read Paddle Models from IR

After [Converting a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md) to [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md), it can be read as recommended. Example:

```python
ie = IECore()
net = ie.read_network("model.xml")
```

## Read Paddle Models from Paddle Format (Paddle `inference model` model type)

**Example:**

```python
ie = IECore()
net = ie.read_network("model.pdmodel")
```

**Reshape feature:**

OpenVINO™ does not provide a mechanism to specify pre-processing, such as mean values subtraction and reverse input channels, for the Paddle format.
If a Paddle model contains dynamic shapes for input, use the [IENetwork.reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a) method for shape specialization.

## NOTE

* A Paddle [`inference model`](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/inference_en.md) mainly contains two kinds of files `model.pdmodel`(model file) and `model.pdiparams`(params file), which are used for inference.
* Supported Paddle models list and how to export these models are described in [Convert a Paddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md).
* For `Normalize` Paddle Models, the input data should be in FP32 format.
* When reading Paddle models from Paddle format, make sure that `model.pdmodel` and `model.pdiparams` are in the same folder directory.
