  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)

  # In our case, using "AUTO" (which should be loaded into the args.device variable)
  exec_net = ie.load_network(network=net, device_name="AUTO", num_requests=num_of_input)
  
  # the following 2 lines are equivalent, alternate ways to do the above
  exec_net = ie.load_network(network=net, device_name="AUTO:CPU,GPU")
  exec_net = ie.load_network(network=net, device_name='"AUTO", {{"AUTO_DEVICE_LIST", "CPU,GPU"}}') 
