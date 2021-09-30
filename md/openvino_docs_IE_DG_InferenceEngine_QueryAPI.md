# Introduction to Inference Engine Device Query API

This section provides a high-level description of the process of querying of different device properties and configuration values. Refer to the Hello Query Device Sample sources and Multi-Device Plugin guide for example of using the Inference Engine Query API in user applications.

## Using the Inference Engine Query API in Your Code

The Inference Engine Core class provides the following API to query device information, set or get different device configuration properties:

* InferenceEngine::Core::GetAvailableDevices - Provides a list of available devices. If there are more than one instance of a specific device, the devices are enumerated with .suffix where suffix is a unique string identifier. The device name can be passed to all methods of the InferenceEngine::Core class that work with devices, for example InferenceEngine::Core::LoadNetwork.
* InferenceEngine::Core::GetMetric - Provides information about specific device. InferenceEngine::Core::GetConfig - Gets the current value of a specific configuration key.
* InferenceEngine::Core::SetConfig - Sets a new value for the configuration key.

The InferenceEngine::ExecutableNetwork class is also extended to support the Query API:
* InferenceEngine::ExecutableNetwork::GetMetric
* InferenceEngine::ExecutableNetwork::GetConfig
* InferenceEngine::ExecutableNetwork::SetConfig

## Query API in the Core Class

### GetAvailableDevices

<pre><code>
  InferenceEngine::Core core;
  std::vector<std::string> availableDevices = core.GetAvailableDevices();
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

### GetConfig

The code below demonstrates how to understand whether HETERO device dumps .dot files with split graphs during the split stage:

<pre><code>
  bool dumpDotFile = core.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>();
</code></pre>

For documentation about common configuration keys, refer to ie_plugin_config.hpp. Device specific configuration keys can be found in corresponding plugin folders.

### GetMetric

To extract device properties such as available device, device name, supported configuration keys, and others, use the InferenceEngine::Core::GetMetric method:

<pre><code>
  std::string cpuDeviceName = core.GetMetric("GPU", METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
</code></pre>

A returned value appears as follows: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz.

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

### SetConfig

The only device that supports this method is the Multi-Device.



