# Multi-Device Plugin

## Introducing the Multi-Device Plugin

The Multi-Device plugin automatically assigns inference requests to available computational devices to execute the requests in parallel. By contrast, the Heterogeneous plugin can run different layers on different devices but not in parallel. The potential gains with the Multi-Device plugin are:

* Improved throughput from using multiple devices (compared to single-device execution)
* More consistent performance, since the devices share the inference burden (if one device is too busy, another can take more of the load)

Note that with Multi-Device the application logic is left unchanged, so you don’t need to explicitly load the network to every device, create and balance the inference requests and so on. From the application point of view, this is just another device that handles the actual machinery. The only thing that is required to leverage performance is to provide the multi-device (and hence the underlying devices) with enough inference requests to process. For example, if you were processing 4 cameras on the CPU (with 4 inference requests), it might be desirable to process more cameras (with more requests in flight) to keep CPU and GPU busy via Multi-Device.

The setup of Multi-Device can be described in three major steps:

1. Configure each device as usual (e.g., via the conventional set_config method)
2. Load the network to the Multi-Device plugin created on top of a (prioritized) list of the configured devices. This is the only change needed in the application.
3. As with any other ExecutableNetwork call (resulting from `load_network`), you create as many requests as needed to saturate the devices. 

These steps are covered below in detail.

## Defining and Configuring the Multi-Device Plugin

Following the OpenVINO™ convention of labeling devices, the Multi-Device plugin uses the name “MULTI”. The only configuration option for the Multi-Device plugin is a prioritized list of devices to use:

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

You can set the configuration directly as a string, or use MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES from the multi/multi_device_config.hpp file, which defines the same string.

### The Three Ways to Specify Devices Targets for the MULTI plugin.

#### Option 1 - Pass a Prioritized List as a Parameter in ie.load_network()

```python
  from openvino.inference_engine import IECore, StatusCode

  # Init the Inference Engine Core
  ie = IECore()

  # Read a network in IR or ONNX format
  net = ie.read_network(model=args.model)
  
  ie.set_config({{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}}, "MULTI");
  
  exec_net_1 = ie.load_network(network=net, device_name=""MULTI", {{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}})
  exec_net_2 = ie.load_network(network=net, device_name=""MULTI:HDDL,GPU")
```
#### Option 2 - Pass a List as a Parameter, and Dynamically Change Priorities during Execution

Notice that the priorities of the devices can be changed in real time for the executable network:

```python
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
```

#### Option 3 - Use Explicit Hints for Controlling Request Numbers Executed by Devices

There is a way to specify the number of requests that Multi-Device will internally keep for each device. If the original app was running 4 cameras with 4 inference requests, it might be best to share these 4 requests between 2 devices used in the MULTI. The easiest way is to specify a number of requests for each device using parentheses: “MULTI:CPU(2),GPU(2)” and use the same 4 requests in the app. However, such an explicit configuration is not performance-portable and not recommended. The better way is to configure the individual devices and query the resulting number of requests to be used at the application level. See [Configuring the Individual Devices and Creating the Multi-Device On Top](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html#configuring_the_individual_devices_and_creating_the_multi_device_on_top).


## Enumerating Available Devices
The Inference Engine features a dedicated API to enumerate devices and their capabilities. See the [Hello Query Device Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html). This is example output from the sample (truncated to device names only):

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


## Configuring the Individual Devices and Creating the Multi-Device On Top

As discussed in the first section, configure each individual device as usual and then create the “MULTI” device on top:

```python
  from openvino.inference_engine import IECore, StatusCode

  ie = IECore()
  
  # Configure the HDDL device first
  net = ie.read_network(model="sample.xml")
  ie.set_config(hddl_config, "HDDL")
  ie.set_config(gpu_config, "GPU")
  
  # Load the network to the multi-device, while specifying the configuration (devices along with priorities):
  exec_net = ie.load_network(network=net, device_name="MULTI", {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "HDDL,GPU"}})

  # A new metric allows querying the optimal number of requests:
  nireq = exec_net.get_metric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
```

An alternative is to combine all the individual device settings into a single config file and load that, allowing the Multi-Device plugin to parse and apply settings to the right devices. See the code example in the next section.

Note that while the performance of accelerators works well with Multi-Device, the CPU+GPU execution poses some performance caveats, as these devices share power, bandwidth and other resources. For example it is recommended to enable the GPU throttling hint (which saves another CPU thread for CPU inferencing). See the section below titled Using the Multi-Device with OpenVINO Samples and Benchmarking the Performance.

## Querying the Optimal Number of Inference Requests

TBD

## Using the Multi-Device with OpenVINO Samples and Benchmarking the Performance

Notice that every OpenVINO sample that supports the `-d` (which stands for “device”) command-line option transparently accepts Multi-Device. The [Benchmark application](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_benchmark_tool_README.html) is the best reference for the optimal usage of Multi-Device. As discussed earlier, you don’t need to set up the number of requests, CPU streams or threads because the application provides optimal performance out of the box. Below is an example command to evaluate HDDL+GPU performance with the Benchmark application:

`./benchmark_app.py –d MULTI:HDDL,GPU –m <model> -i <input> -niter 1000`

Notice that you can use the FP16 IR to work with Multi-Device, as CPU automatically upconverts it to FP32 and the other devices support it natively. Also notice that no demos are (yet) fully optimized for Multi-Device, by means of supporting the OPTIMAL_NUMBER_OF_INFER_REQUESTS metric, using the GPU streams/throttling, and so on.

## Video: MULTI Plugin
NOTE: This video is currently available only for C++, but many of the same concepts apply to Python.

[![MULTI Plugin YouTube Tutorial](https://img.youtube.com/vi/xbORYFEmrqU/0.jpg)](https://www.youtube.com/watch?v=xbORYFEmrqU)


See Also:
[Supported Devices](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
