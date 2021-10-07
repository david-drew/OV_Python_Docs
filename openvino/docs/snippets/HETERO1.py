from openvino.inference_engine import IECore
import ngraph as ng

# Set up the network as usual, then pass it as a parameter
ie = IECore()
net = ie.read_network(model='sample.xml')

# Now we can extract an Ngraph function
func = ng.function_from_cnn(net)

# This example demonstrates how to perform default affinity initialization and then
# correct affinity manually for some layers
device_target = "HETERO:GPU,CPU"

# Query Network result object maps layer -> device
res = ie.query_network(network=net, device_name=device_target)

# Each layer/node could be a different value if desired
for r in res:
    res[r] = 'CPU'

# Now we can find and change a given node's affinity
for node in func.get_ordered_ops():
  rt_info = node.get_rt_info()
  rt_info['affinity'] = 'CPU'

exec_net = ie.load_network(network=net, device_name=device_target)
