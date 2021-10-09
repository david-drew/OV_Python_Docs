# Introduction to Inference Engine Device Query API

The OpenVINO™ toolkit supports inferencing with several types of devices (processors or accelerators). 
This section provides a high-level description of the process of querying device properties and configuration values at runtime. Refer to the [Hello Query Device Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html) sources and the [Multi-Device Plugin guide](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html) for example of using the Inference Engine Query API in user applications.

## Using the Inference Engine Query API in Your Code

The [Inference Engine Core](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1IECore.html) class provides the following API to query device information, set or get different device configuration properties:

* [ie_api.IECore.available_devices](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1IECore.html#a53ae93f362e9ceb7ffe27fcd20000025) - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with .suffix where suffix is a unique string identifier. The device name can be passed to all methods of the IECore class that work with devices, for example [ie_api.IECore.load_network](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1IECore.html#ac9a2e043d14ccfa9c6bbf626cfd69fcc).
* [ie_api.ieCore.get_metric](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#af1cdf2ecbea6399c556957c2c2fdf8eb) - Provides information about specific device.
* [ie_api.IECore.get_config](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a48764dec7c235d2374af8b8ef53c6363) - Gets the current value of a specific configuration key.
* [ie_api.IECore.set_config](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a2c738cee90fca27146e629825c039a05)  - Sets a new value for the configuration key.

The [ie_api.ExecutableNetwork](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1ExecutableNetwork.html) class is also extended to support the Query API:
* [ie_api.ExecutableNetwork.get_metric](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1ExecutableNetwork.html#ab1266563989479fd897250390f4ca23b)
* [ie_api.ExecutableNetwork.get_config](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1ExecutableNetwork.html#a41880d0a92e9f34096f38b81b0fef3db)
* There is no method for `set_config` corresponding to `get_config`, but the equivalent action is described below.

## Query API in the Core Class

### GetAvailableDevices

```python
  from openvino.inference_engine import IECore, StatusCode
  ie = IECore()
  available_devices = ie.get_available_devices()
```

The function returns a list of available devices, for example:

<pre><code>  MYRIAD.1.2-ma2480
  MYRIAD.1.4-ma2480
  FPGA.0
  FPGA.1
  CPU
  GPU.0
  GPU.1
</code></pre>

Each device name can then be passed to:
- IECore.load_network to load the network to a specific device.
- IECore.get_metric to get common or device specific metrics.
- All other methods of the Core class that accept deviceName.

### [GetConfig](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a48764dec7c235d2374af8b8ef53c6363)

The code below demonstrates how to understand whether HETERO device dumps .dot files with split graphs during the split stage:

<pre><code>  ie = IECore()
  ie.get_metric(metric_name="DUMP_GRAPH_DOT", device_name="HETERO")
</code></pre>

For documentation about common configuration keys, refer to ie_plugin_config.hpp. Device specific configuration keys can be found in corresponding plugin folders.

### [GetMetric](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#af1cdf2ecbea6399c556957c2c2fdf8eb)

To extract device properties such as available device, device name, supported configuration keys, and others, use the InferenceEngine::Core::GetMetric method:

```python
  ie = IECore()
  ie.get_metric(metric_name="FULL_DEVICE_NAME", device_name="GPU")
```

A returned value appears as follows: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz.

To list all supported metrics:

```python
  ie = IECore()
  ie.get_metric(metric_name="SUPPORTED_METRICS", device_name="GPU")
```


**NOTE:** All metrics have a specific type, which is set during the metric instantiation. The list of common device-agnostic metrics can be found in ie_plugin_config.hpp. Device specific metrics (for example, for HDDL, MYRIAD devices) can be found in corresponding plugin folders.

## Query API in the ExecutableNetwork Class

### [GetMetric](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1ExecutableNetwork.html#ab1266563989479fd897250390f4ca23b)

The method is used to get an executable network specific metric such as the network name running on a device:

```python
  ie = IECore()
  net = ie.read_network(model="sample.xml", weights="sample.bin")
  exec_net = ie.load_network(network=net, device_name="CPU")
  exec_net.get_metric(metric_name="NETWORK_NAME", device_name="CPU") 
```

Or the current temperature of MYRIAD device:

```python
  ie = IECore()
  net = ie.read_network(model="sample.xml", weights="sample.bin")
  exec_net = ie.load_network(network=net, device_name="MYRIAD")
  num_threads = exec_net.get_metric("DEVICE_THERMAL")
```

### [GetConfig](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1ExecutableNetwork.html#a41880d0a92e9f34096f38b81b0fef3db)
The method is used to get information about configuration values the executable network has been created with:

```python
  ie = IECore()
  net = ie.read_network(model="sample.xml", weights="sample.bin")
  exec_net = ie.load_network(network=net, device_name="CPU")
  num_threads = exec_net.get_config(metric_name="KEY_CPU_THREADS_NUM", device_name="CPU")
```

### [SetConfig](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a2c738cee90fca27146e629825c039a05)

The only device that supports this method is the [Multi-Device](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html).

To set a configuration, a valid key must be used to create a dictionary, which is passed as the 'config' parameter to the IECore.load_network call.
```python
  ie = IECore()
  bf16_config = {"ENFORCE_BF16" : "YES"}
  exec_net = ie.load_network(network=net, device_name="MULTI", config=bf16_config)
```
