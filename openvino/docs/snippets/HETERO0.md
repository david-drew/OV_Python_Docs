from openvino.inference_engine import IECore
import ngraph as ng

# Set up the network as usual, then pass it as a parameter
ie = IECore()
net = ie.read_network(model='sample.xml')

# Now we can extract an Ngraph function
func = ng.function_from_cnn(net)

# Now we can find and change a given node's affinity
for node in func.get_ordered_ops():
  rt_info = node.get_rt_info()
  rt_info['affinity'] = 'CPU'
