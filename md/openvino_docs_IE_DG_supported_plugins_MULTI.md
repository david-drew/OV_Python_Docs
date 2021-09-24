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

Following the OpenVINO convention of labeling “devices”, the Multi-Device has a “MULTI” name. The only configuration option for the Multi-Device plugin is a prioritized list of devices to use:

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






See Also:
[Supported Devices](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
