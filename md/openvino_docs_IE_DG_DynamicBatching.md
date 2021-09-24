# How to Use Dynamic Batching

Dynamic Batching is a feature that allows you to dynamically change batch size for inference calls within a preset batch size limit. This feature might be useful when batch size is unknown beforehand, and using extra large batch size is not desired or impossible due to resource limitations. For example, face detection with person age, gender, or mood recognition is a typical usage scenario.

You can activate Dynamic Batching by setting KEY_DYN_BATCH_ENABLED flag to YES in a configuration map that is passed to the plugin while loading a network. This configuration creates an ExecutableNetwork object that will allow setting batch size dynamically in all of its infer requests using SetBatch() method. The batch size that was set in passed CNNNetwork object will be used as a maximum batch size limit.

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Get the dynamic batch size from the command line parameter
  dyn_batch_size = args.dyn_bs

  dyn_config = {"DYN_BATCH_ENABLED": "YES"}

  # Init the Inference Engine Core
  ie = IECore()
  
  ie.set_config(dyn_config)
    
  # Read a network in IR or ONNX format
  net = ie.read_network("sample.xml")
    
  # Enable dynamic batching and prepare for setting max batch limit
  net.set_batch_size = int(arguments.ag_max_batch_size)

  # Create executable network
  exec_net = ie.load_network(network=net, device_name=args.device, config=dyn_config)

  # Usually this will be looped until the end of the video, list of images, etc.
  
  # Most of the OpenCV and related code is not included here
  have_frame, frame = capture.read()

  while have_frame:
    # Process frames (images or video frames)
    # Dynamically set batch size for subsequent Infer() calls of this request
    batch_size = images_data.size();
    xec_net.set_batch(batch_size);
    
    # Run inference
    results = exec_net.infer();

    # Will end if out of input
    have_frame, frame = capture.read()
</code></pre>

Limitations
Currently, certain limitations for using Dynamic Batching exist:

* Use Dynamic Batching with CPU and GPU plugins only.
* Use Dynamic Batching on topologies that consist of certain layers only:
  * Convolution
  * Deconvolution
  * Activation
  * LRN
  * Pooling
  * FullyConnected
  * SoftMax
  * Split
  * Concatenation
  * Power
  * Eltwise
  * Crop
  * BatchNormalization
  * Copy

Do not use layers that might arbitrary change tensor shape (such as Flatten, Permute, Reshape), layers specific to object detection topologies (ROIPooling, ProirBox, DetectionOutput), and custom layers. Topology analysis is performed during the process of loading a network into plugin, and if topology is not applicable, an exception is generated.
