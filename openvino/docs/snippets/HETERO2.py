ie = IECore()
net = ie.read_network('sample.xml')
exec_net = ie.load_network(network=net, device_name='HETERO:GPU,CPU')
