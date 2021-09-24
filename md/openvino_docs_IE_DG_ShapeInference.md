# Under Construction

*Really: Under Construction*

# Using Shape Inference

OpenVINO™ provides the following methods for runtime model reshaping:

* Set a new input shape with the InferenceEngine::CNNNetwork::reshape method.

The [IENetwork::reshape](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a6683f0291db25f908f8d6720ab2f221a) method updates input shapes and propagates them down to the outputs of the model through all intermediate layers.



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

See Also:
The [Hello Reshape Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_reshape_ssd_README.html)



