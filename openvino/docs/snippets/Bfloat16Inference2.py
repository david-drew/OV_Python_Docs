from openvino.inference_engine import IECore, StatusCode

bf16_config = {"ENFORCE_BF16" : "YES"}

ie = IECore()
net = ie.read_network("sample.xml")
exec_net = ie.load_network(network=net, device_name="CPU", config = bf16_config)
