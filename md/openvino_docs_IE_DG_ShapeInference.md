# Under Construction

*Really: Under Construction*

# Using Shape Inference

OpenVINO™ provides the following methods for runtime model reshaping:

* Set a new input shape with the [IENetwork::reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a) method.

The [IENetwork::reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a) method updates input shapes and propagates them down to the outputs of the model through all intermediate layers.

## NOTES

* Starting with the 2021.1 release, the Model Optimizer converts topologies keeping shape-calculating sub-graphs by default, which enables correct shape propagation during reshaping in most cases.
* Older versions of IRs are not guaranteed to reshape successfully. Please regenerate them with the Model Optimizer of the latest version of OpenVINO™.
* If an ONNX model does not have a fully defined input shape and the model was imported with the ONNX importer, reshape the model before loading it to the plugin.
* Set a new batch dimension value with the InferenceEngine::CNNNetwork::setBatchSize method.

  The meaning of a model batch may vary depending on the model design. This method does not deduce batch placement for inputs from the model architecture. It assumes that the batch is placed at the zero index in the shape for all inputs and uses the InferenceEngine::CNNNetwork::reshape method to propagate updated shapes through the model.

  The method transforms the model before a new shape propagation to relax a hard-coded batch dimension in the model, if any.

  Use [IENetwork::reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a) rather than  [IENetwork::batch_size](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a79a647cb1b49645616eaeb2ca255ef2e) to set new input shapes for the model if the model has:

    * Multiple inputs with different zero-index dimension meanings
    * Input without a batch dimension
    * 0D, 1D, or 3D shape

  The [IENetwork::batch_size](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a79a647cb1b49645616eaeb2ca255ef2e) method is a high-level API method that wraps the [IENetwork::reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a)  method call and works for trivial models from the batch placement standpoint. Use InferenceEngine::CNNNetwork::reshape for other models.

  Using the [IENetwork::batch_size](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a79a647cb1b49645616eaeb2ca255ef2e) method for models with a non-zero index batch placement or for models with inputs that do not have a batch dimension may lead to undefined behaviour.

You can change input shapes multiple times using the IENetwork::reshape and IENetwork::setBatchSize methods in any order. If a model has a hard-coded batch dimension, use IENetwork::setBatchSize first to change the batch, then call IENetwork::reshape to update other dimensions, if needed.

Inference Engine takes three kinds of a model description as an input, which are converted into an IENetwork object:

1. Intermediate Representation (IR) through IECore::ReadNetwork
2. ONNX model through IECore::ReadNetwork
3. nGraph function through the constructor of IENetwork

IENetwork keeps an ngraph::Function object with the model description internally. The object should have fully defined input shapes to be successfully loaded to the Inference Engine plugins. To resolve undefined input dimensions of a model, call the IENetwork::reshape method providing new input shapes before loading to the Inference Engine plugin.

Run the following code right after IENetwork creation to explicitly check for model input names and shapes:

To feed input data of a shape that is different from the model input shape, reshape the model first.

Once the input shape of IENetwork is set, call the IECore::LoadNetwork method to get an InferenceEngine::ExecutableNetwork object for inference with updated shapes.

There are other approaches to reshape the model during the stage of IR generation or nGraph::Function creation.

Practically, some models are not ready to be reshaped. In this case, a new input shape cannot be set with the Model Optimizer or the IENetwork::reshape method.

## Troubleshooting Reshape Errors
Operation semantics may impose restrictions on input shapes of the operation. Shape collision during shape propagation may be a sign that a new shape does not satisfy the restrictions. Changing the model input shape may result in intermediate operations shape collision.

Examples of such operations:

* Reshape operation with a hard-coded output shape value
* MatMul operation with the Const second input cannot be resized by spatial dimensions due to operation semantics

A model's structure and logic should not significantly change after model reshaping.

* The Global Pooling operation is commonly used to reduce output feature map of classification models output. Having the input of the shape [N, C, H, W], Global Pooling returns the output of the shape [N, C, 1, 1]. Model architects usually express Global Pooling with the help of the Pooling operation with the fixed kernel size [H, W]. During spatial reshape, having the input of the shape [N, C, H1, W1], Pooling with the fixed kernel size [H, W] returns the output of the shape [N, C, H2, W2], where H2 and W2 are commonly not equal to 1. It breaks the classification model structure. For example, publicly available Inception family models from TensorFlow* have this issue.

* Changing the model input shape may significantly affect its accuracy. For example, Object Detection models from TensorFlow have resizing restrictions by design. To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the pipeline.config file. For details, refer to the Tensorflow Object Detection API models resizing techniques.


## Usage of the Reshape Method
The primary method of the feature is IENetwork::reshape. It gets new input shapes and propagates it from input to output for all intermediates layers of the given network. The method takes IENetwork::InputShapes - a map of pairs: name of input data and its dimension.

The algorithm for resizing network is the following:

1. Collect the map of input names and shapes from Intermediate Representation (IR) using helper method IENetwork::getInputShapes
2. Set new input shapes
3. Call reshape

Here is a code example:

<pre><code>
  # Libraries needed for this example
  import cv2
  import numpy as np
  from openvino.inference_engine import IECore, StatusCode

  ie = IECore()

  # 0. Read IR and image
  #--------------------------------------------------------------------------------
  net = ie.read_network(model="sample.xml")
  image_orig = cv2.imread("/path/to/image")
  image = image_orig.copy()

  # 1. Get the information needed for the reshape
  #--------------------------------------------------------------------------------

  # Get names of input and output blobs (for setting precision in next lines)
  input_blob = next(iter(net.input_info))
  out_blob = next(iter(net.outputs))

  # 2. Set new input shapes
  #--------------------------------------------------------------------------------

  # Set input and output precision manually
  net.input_info[input_blob].precision = 'U8'
  net.outputs[out_blob].precision = 'FP32'

  # Change data layout from HWC to CHW
  image = image.transpose((2, 0, 1))

  # Add N dimension to transform to NCHW
  image = np.expand_dims(image, axis=0)

  # 3. Call reshape
  #--------------------------------------------------------------------------------
  net.reshape({input_blob: image.shape})

  # 4. Load the model to the device and proceed with inference
  exec_net = ie.load_network(network=net, device_name="CPU")

  # etc...
</code></pre>

## Extensibility
The Inference Engine provides a special mechanism that allows adding support of shape inference for custom operations. This mechanism is described in the [Extensibility documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Extensibility_DG_Intro.html)

See Also:

* The [Hello Reshape Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README.html)
* The [C++ Smart Classroom Demo](https://docs.openvinotoolkit.org/latest/omz_demos_smart_classroom_demo_cpp.html) is another sample demonstrating shape inference features.



