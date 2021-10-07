from openvino.inference_engine import IECore, StatusCode

# Init the Inference Engine Core
ie = IECore()

# Read a network in IR or ONNX format
net = ie.read_network(model="sample.xml")

# Load the network to the AUTO device
exec_net = ie.load_network(network=net, device_name="AUTO", config=full_config)
  
# Query the device's optimization capabilities
device_caps = ie.get_metric('OPTIMIZATION_CAPABILITIES')
