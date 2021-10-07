  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)

  # Load a network on the target device
  exec_net = ie.load_network(network=net, device_name=args.device, num_requests=num_of_input)
  
  # In our case, using "AUTO" (which should be loaded into the args.device variable)
  exec_net = ie.load_network(network=net, device_name="AUTO", num_requests=num_of_input)
