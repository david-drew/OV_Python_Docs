

# Auto-Device Plugin

## Auto-Device Plugin Execution
The AUTO device is a new, special “virtual” or “proxy” device in the OpenVINO™ toolkit.

Use “AUTO” as the device name to delegate selection of an actual accelerator to OpenVINO. With the 2021.4 release, the Auto-device plugin internally recognizes and selects devices from among CPU, integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristics of CNN models (for example, precision). Then the Auto-device assigns inference requests to the selected device.

From the application point of view, this is just another device that handles all accelerators in the full system.

With the 2021.4 release, Auto-device setup is done in three major steps:

1. Configure each device as usual (for example, via the conventional SetConfig method).
2. Load a network to the Auto-device plugin. This is the only change needed in your application.
3. Just like with any other executable network (resulting from LoadNetwork), create as many requests as needed to saturate the devices. These steps are covered below in detail.

## Defining and Configuring the Auto-Device Plugin
Following the OpenVINO convention for devices names, the Auto-device uses the label “AUTO”. The only configuration option for Auto-device is a limited device list:

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

There are two ways to use the Auto-device plugin:

1. Directly indicate device by “AUTO” or an empty string.
```python
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)

  # Load a network on the target device
  exec_net = ie.load_network(network=net, device_name=args.device, num_requests=num_of_input)
  
  # In our case, using "AUTO" (which should be loaded into the args.device variable)
  exec_net = ie.load_network(network=net, device_name="AUTO", num_requests=num_of_input)
```

2. Or use the Auto-device configuration to limit the device candidates list to be selected.
```python
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
```

The Auto-device plugin supports query device optimization capabilities in metric.

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

The Inference Engine now features a dedicated API to enumerate devices and their capabilities. See the [Hello Query Device Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html) for code.

This is the example output from the sample (truncated to device names only):

```python
./hello_query_device

Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```

### Enumerating Available Devices

### Default Auto-Device Selection Logic
With the 2021.4 release, Auto-Device selects the most suitable device with following default logic:
1. Check if dGPU (discrete), iGPU (integrated) and CPU devices are available
2. Get the precision of the input model, such as FP32
3. According to the priority of dGPU, iGPU, and CPU (in this order), if the device supports the precision of the input network, select it as the most suitable device

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

* When the application uses Auto-device to run FP16 IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to dGPU.
* When the application uses Auto-device to run FP16 IR on a system with CPU and iGPU, Auto-device will offload this workload to iGPU.
* When the application uses Auto-device to run WINOGRAD-enabled IR on a system with CPU, dGPU and iGPU, Auto-device will offload this workload to CPU.

In any case, when loading the network to dGPU or iGPU fails, the networks uses CPU as the fall-back choice.

### Limit Auto Target Devices Logic

According to the Auto-device selection logic from the previous section, the most suitable device from available devices to load mode as follows:

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model="sample.xml")

  # Load the network to the AUTO device
  exec_net = ie.load_network(network=net, device_name="AUTO")
</code></pre>

Another way to load mode to device from limited choice of devices is with Auto-device:

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model="sample.xml")

  # Load the network to the AUTO device
  exec_net = ie.load_network(network=net, device_name="AUTO:CPU,GPU")
</code></pre>

## Configuring the Individual Devices and Creating the Auto-Device on Top
As described in the first section, configure each individual device as usual and then just create the “AUTO” device on top:

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model="sample.xml")
  
  ie.set_config(cpu_config, "CPU")
  ie.set_config(gpu_config, "GPU")

  # Load the network to the AUTO device
  exec_net = ie.load_network(network=net, device_name="AUTO")
  
  # Query the device's optimization capabilities
  device_caps = exec_net.get_metric(METRIC_KEY(OPTIMIZATION_CAPABILITIES))
</code></pre>


Alternatively, you can combine all the individual device settings into single config file and load it, allowing the Auto-device plugin to parse and apply it to the right devices. See the code example here:

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model="sample.xml")

  # Load the network to the AUTO device
  exec_net = ie.load_network(network=net, device_name="AUTO", config=full_config)
  
  # Query the device's optimization capabilities
  device_caps = exec_net.get_metric(METRIC_KEY(OPTIMIZATION_CAPABILITIES))
</code></pre>


## Using the Auto-Device with OpenVINO Samples and Benchmark App

Note that every OpenVINO sample that supports the “-d” (which stands for “device”) command-line option transparently accepts the Auto-device. The Benchmark Application is the best example of the optimal usage of the Auto-device. You do not need to set the number of requests and CPU threads, as the application provides optimal out-of-the-box performance. Below is the example command-line to evaluate AUTO performance with that:

`./benchmark_app.py –d AUTO –m <model> -i <input> -niter 1000`

You can also use the auto-device with limit device choice:

`./benchmark_app.py –d AUTO:CPU,GPU –m <model> -i <input> -niter 1000`
