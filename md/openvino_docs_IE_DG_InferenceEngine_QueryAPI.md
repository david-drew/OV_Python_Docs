# Introduction to Inference Engine Device Query API

This section provides a high-level description of the process of querying of different device properties and configuration values. Refer to the [Hello Query Device Python Sample](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README.html) sources and the [Multi-Device Plugin guide](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html) for example of using the Inference Engine Query API in user applications.

## Using the Inference Engine Query API in Your Code

The Inference Engine Core class provides the following API to query device information, set or get different device configuration properties:

* InferenceEngine::Core::GetAvailableDevices - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with .suffix where suffix is a unique string identifier. The device name can be passed to all methods of the InferenceEngine::Core class that work with devices, for example InferenceEngine::Core::LoadNetwork.
* InferenceEngine::Core::GetMetric - Provides information about specific device. InferenceEngine::Core::GetConfig - Gets the current value of a specific configuration key.
* InferenceEngine::Core::SetConfig - Sets a new value for the configuration key.

The InferenceEngine::ExecutableNetwork class is also extended to support the Query API:
* [ie_api.get_metric](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#af1cdf2ecbea6399c556957c2c2fdf8eb)
* [ie_api.get_config](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a48764dec7c235d2374af8b8ef53c6363)
* [ie_api.set_config](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a2c738cee90fca27146e629825c039a05)

## Query API in the Core Class

### GetAvailableDevices

<pre><code>
  from openvino.inference_engine import IECore, StatusCode
  ie = IECore()
  available_devices = ie.get_available_devices()
</code></pre>

The function returns a list of available devices, for example:

<pre><code>
  MYRIAD.1.2-ma2480
  MYRIAD.1.4-ma2480
  FPGA.0
  FPGA.1
  CPU
  GPU.0
  GPU.1
</code></pre>

Each device name can then be passed to:
- InferenceEngine::Core::LoadNetwork to load the network to a specific device.
- InferenceEngine::Core::GetMetric to get common or device specific metrics.
- All other methods of the Core class that accept deviceName.

### [GetConfig](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a48764dec7c235d2374af8b8ef53c6363)

The code below demonstrates how to understand whether HETERO device dumps .dot files with split graphs during the split stage:

<pre><code>
  dump_dot_file = ie.get_config(device_name="HETERO", config_name="DUMP_GRAPH_DOT")
</code></pre>

For documentation about common configuration keys, refer to ie_plugin_config.hpp. Device specific configuration keys can be found in corresponding plugin folders.

### [GetMetric](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#af1cdf2ecbea6399c556957c2c2fdf8eb)

To extract device properties such as available device, device name, supported configuration keys, and others, use the InferenceEngine::Core::GetMetric method:

<pre><code>
  ie = IECore()
  ie.get_metric(metric_name="FULL_DEVICE_NAME", device_name="GPU")
</code></pre>

A returned value appears as follows: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz.

To list all supported metrics:

<pre><code>
  ie = IECore()
  ie.get_metric(metric_name="SUPPORTED_METRICS", device_name="GPU")
</code></pre>


**NOTE:** All metrics have a specific type, which is set during the metric instantiation. The list of common device-agnostic metrics can be found in ie_plugin_config.hpp. Device specific metrics (for example, for HDDL, MYRIAD devices) can be found in corresponding plugin folders.

## Query API in the ExecutableNetwork Class

### GetMetric

<pre><code>
  auto network = core.ReadNetwork("sample.xml");
  auto exeNetwork = core.LoadNetwork(network, "CPU");
  auto nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
</code></pre>

### GetConfig
The method is used to get information about configuration values the executable network has been created with:

<pre><code>
  auto network = core.ReadNetwork("sample.xml");
  auto exeNetwork = core.LoadNetwork(network, "CPU");
  auto ncores = exeNetwork.GetConfig(PluginConfigParams::KEY_CPU_THREADS_NUM).as<std::string>();
</code></pre>

Or the current temperature of MYRIAD device:

<pre><code>
  auto network = core.ReadNetwork("sample.xml");
  auto exeNetwork = core.LoadNetwork(network, "MYRIAD");
  float temperature = exeNetwork.GetMetric(METRIC_KEY(DEVICE_THERMAL)).as<float>();
</code></pre>

### [SetConfig](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a2c738cee90fca27146e629825c039a05)

The only device that supports this method is the (Multi-Device)[https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html].



