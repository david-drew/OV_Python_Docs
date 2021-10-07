from openvino.inference_engine import IECore, StatusCode
  
# Get the dynamic batch size from the command line parameter - must be an int
dyn_batch_size = 1
  
if isinstance(args.dyn_bs, int):
  dyn_batch_size = args.dyn_bs

# Setup our configuration map
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
  exec_net.set_batch(batch_size);
    
  # Run inference
  results = exec_net.infer();

  # Will end if out of input
  have_frame, frame = capture.read()
