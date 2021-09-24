# Multi-Device Plugin

## Introducing the Multi-Device Plugin

The Multi-Device plugin automatically assigns inference requests to available computational devices to execute the requests in parallel. The potential gains are:

* Improved throughput that multiple devices can deliver (compared to single-device execution)
* More consistent performance, since the devices share the inference burden (if one device is too busy, another can take more of the load)

Note that with multi-device the application logic is left unchanged, so you don’t need to explicitly load the network to every device, create and balance the inference requests and so on. From the application point of view, this is just another device that handles the actual machinery. The only thing that is required to leverage performance is to provide the multi-device (and hence the underlying devices) with enough inference requests to process. For example, if you were processing 4 cameras on the CPU (with 4 inference requests), it might be desirable to process more cameras (with more requests in flight) to keep CPU+GPU busy via multi-device.

The “setup” of Multi-Device can be described in three major steps:

1. Configure each device as usual (e.g. via the conventional set_config method)
2. Load the network to the Multi-Device plugin created on top of a (prioritized) list of the configured devices. This is the only change needed in the application.
3. As with any other ExecutableNetwork (resulted from `load_network`) you create as many requests as needed to saturate the devices. 

These steps are covered below in detail.

## Step 1 - Defining and Configuring the Multi-Device Plugin

Following the OpenVINO™ convention of labeling “devices”, the Multi-Device has a “MULTI” name. The only configuration option for the Multi-Device plugin is a prioritized list of devices to use:

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
<tr class="row-even"><td><p>“MULTI_DEVICE_PRIORITIES”</p></td>
<td><p>comma-separated device names with no spaces</p></td>
<td><p>N/A</p></td>
<td><p>Prioritized list of devices</p></td>
</tr>
</tbody>
</table>

You can set the configuration directly as a string, or use MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES from the multi/multi_device_config.hpp, which defines the same string.

There are three ways to specify the devices to be use by the “MULTI”:

### Enumerating Available Devices
The Inference Engine features a dedicated API to enumerate devices and their capabilities. See the [Hello Query Device Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html). This is example output from the sample (truncated to the devices’ names only):

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)
  
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}}, "MULTI");
  
  exec_net_1 = ie.load_network(network=net, device_name=""MULTI", {{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}})
  exec_net_2 = ie.load_network(network=net, device_name=""MULTI:HDDL,GPU")
</code></pre>

Notice that the priorities of the devices can be changed in real time for the executable network:

<pre><code>
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)
  
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}}, "MULTI")
  
  # Change priorities
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "GPU,HDDL"}}, "MULTI")
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "GPU"}}, "MULTI")
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}}, "MULTI")
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "CPU,HDDL,GPU"}}, "MULTI")
</code></pre>

There is a way to specify the number of requests that the multi-device will internally keep for each device. If the original app was running 4 cameras with 4 inference requests, it might be best to share these 4 requests between 2 devices used in the MULTI. The easiest way is to specify a number of requests for each device using parentheses: “MULTI:CPU(2),GPU(2)” and use the same 4 requests in the app. However, such an explicit configuration is not performance-portable and not recommended. The better way is to configure the individual devices and query the resulting number of requests to be used at the application level. See [Configuring the Individual Devices and Creating the Multi-Device On Top](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html#configuring_the_individual_devices_and_creating_the_multi_device_on_top).

 
## Enumerating Available Devices

The Inference Engine features a dedicated API to enumerate devices and their capabilities. See [Hello Query Device Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html). This is example output from the sample (truncated to the devices’ names only):

<pre><code>
./hello_query_device
Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
...
    Device: HDDL
</code></pre>

A simple programmatic way to enumerate the devices and use with the multi-device is as follows:

Beyond the simple device labels such as “CPU”, “GPU”, “HDDL”, when multiple instances of a device are available the names are more qualified. For example, this is how two Intel® Movidius™ Myriad™ X sticks are listed with the hello_query_sample:

<pre><code>
...
    Device: MYRIAD.1.2-ma2480
...
    Device: MYRIAD.1.4-ma2480
</code></pre>



See Also:
[Supported Devices](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
