

# Auto-Device Plugin

## Auto-Device Plugin Execution
The "AUTO" device is a new, special “virtual” or “proxy” device in the OpenVINO™ toolkit.

Use “AUTO” as the device name to delegate selection of an actual accelerator to OpenVINO. With the 2021.4 release, the Auto-device internally recognizes and selects devices from CPU, integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristic of CNN models, for example, precisions. Then Auto-device assigns inference requests to the selected device.

From the application point of view, this is just another device that handles all accelerators in full system.

With the 2021.4 release, Auto-device setup is done in three major steps:

1. Configure each device as usual (for example, via the conventional SetConfig method)
2. Load a network to the Auto-device plugin. This is the only change needed in your application
3. Just like with any other executable network (resulted from LoadNetwork), create as many requests as needed to saturate the devices. These steps are covered below in details.

## Defining and Configuring the Auto-Device Plugin
Following the OpenVINO convention for devices names, the Auto-device uses the label “AUTO”. The only configuration option for Auto-device is a limited device list:
*[HTML]
<table class="table">
<colgroup>
<col style="width: 17%" />
<col style="width: 41%" />
<col style="width: 7%" />
<col style="width: 35%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head"><p>Parameter name</p></th>
<th class="head"><p>Parameter values</p></th>
<th class="head"><p>Default</p></th>
<th class="head"><p>Description</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>“AUTO_DEVICE_LIST”</p></td>
<td><p>comma-separated device names with no spaces</p></td>
<td><p>N/A</p></td>
<td><p>Device candidate list to be selected</p></td>
</tr>
</tbody>
</table>

You can use the configuration name directly as a string or use IE::KEY_AUTO_DEVICE_LIST from ie_plugin_config.hpp, which defines the same string.

There are two ways to use the Auto-device:
1. Directly indicate device by “AUTO” or an empty string.
<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)

  # Load a network on the target device
  exec_net = ie.load_network(network=net, device_name=args.device, num_requests=num_of_input)
  
  # In our case, using "AUTO" (which should be loaded into the args.device variable)
  exec_net = ie.load_network(network=net, device_name="AUTO", num_requests=num_of_input)
</code></pre>

2. Use the Auto-device configuration to limit the device candidates list to be selected
<pre><code>
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
</code></pre>

The Auto-device supports query device optimization capabilities in metric

*[HTML]
<table class="table">
<colgroup>
<col style="width: 53%" />
<col style="width: 47%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head"><p>Parameter name</p></th>
<th class="head"><p>Parameter values</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>“OPTIMIZATION_CAPABILITIES”</p></td>
<td><p>Auto-Device capabilities</p></td>
</tr>
</tbody>
</table>

## Enumerating Devices and Selection Logic

The Inference Engine now features a dedicated API to enumerate devices and their capabilities. See the [Hello Query Device C++ Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_samples_hello_query_device_README.html)

This is the example output from the sample (truncated to the devices’ names only):</p>
<pre class="highlight literal-block"><span></span><span class="p">.</span><span class="o">/</span><span class="n">hello_query_device</span>
<span class="n">Available</span> <span class="nl">devices</span><span class="p">:</span>
    <span class="nl">Device</span><span class="p">:</span> <span class="n">CPU</span>
<span class="p">...</span>
    <span class="nl">Device</span><span class="p">:</span> <span class="n">GPU</span><span class="mf">.0</span>
<span class="p">...</span>
    <span class="nl">Device</span><span class="p">:</span> <span class="n">GPU</span><span class="mf">.1</span></pre>


### Enumerating Available Devices

### Default Auto-Device Selection Logic
With the 2021.4 release, Auto-Device selects the most suitable device with following default logic:
1. Check if dGPU, iGPU and CPU device are available
2. Get the precision of the input model, such as FP32
3. According to the priority of dGPU, iGPU and CPU (in this order), if the device supports the precision of the input network, select it as the most suitable device

For example, CPU, dGPU and iGPU can support the following precision and optimization capabilities:

<table class="table">
<colgroup>
<col style="width: 16%" />
<col style="width: 84%" />
</colgroup>
<thead>
<tr class="row-odd"><th class="head"><p>Device</p></th>
<th class="head"><p>OPTIMIZATION_CAPABILITIES</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>CPU</p></td>
<td><p>WINOGRAD FP32 FP16 INT8 BIN</p></td>
</tr>
<tr class="row-odd"><td><p>dGPU</p></td>
<td><p>FP32 BIN BATCHED_BLOB FP16 INT8</p></td>
</tr>
<tr class="row-even"><td><p>iGPU</p></td>
<td><p>FP32 BIN BATCHED_BLOB FP16 INT8</p></td>
</tr>
</tbody>
</table>

* When application uses Auto-device to run FP16 IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to dGPU.
* When application uses Auto-device to run FP16 IR on a system with CPU and iGPU, Auto-device will offload this workload to iGPU.
* When application uses Auto-device to run WINOGRAD-enabled IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to CPU.

In any case, when loading the network to dGPU or iGPU fails, the networks falls back to CPU as the last choice.

## Limit Auto Target Devices Logic

