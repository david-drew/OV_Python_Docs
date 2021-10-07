from openvino.inference_engine import IECore, StatusCode

ie = IECore()
net = ie.read_network("sample.xml")
cpu_caps = ie.get_metric(metric_name='OPTIMIZATION_CAPABILITIES', device_name='CPU'
