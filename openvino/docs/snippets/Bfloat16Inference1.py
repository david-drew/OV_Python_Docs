from openvino.inference_engine import IECore, StatusCode

ie = IECore()
net = ie.read_network("sample.xml")
exec_net = ie.load_network(network=net, device_name="CPU")
cpu_caps = exec_net.get_config("ENFORCE_BF16")
