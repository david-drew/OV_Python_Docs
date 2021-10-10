# Model Caching Overview

## Introduction

As described in Inference Engine Developer Guide, a common application flow consists of the following steps:

1. **Create an Inference Engine Core Object**
2. **Read the Intermediate Representation** - Read an Intermediate Representation file into an object of the [ie_api.IENetwork](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html)
3. **Prepare inputs and outputs**
4. **Set configuration** - Pass device-specific loading configurations to the device
5. **Compile and Load Network to device** - Use the IECore.load_network() method and specify the target device
6. **Set input data**
7. **Execute the model** - Run inference

Step #5 can potentially perform several time-consuming device-specific optimizations and network compilations, and such delays can lead to bad user experience on application startup. To avoid this, some devices offer Import/Export network capability, and it is possible to either use the [Compile Tool](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_compile_tool_README.html) or enable model caching to export the compiled network automatically. Reusing cached networks can significantly reduce load network time.

## Set the “CACHE_DIR” config option to enable model caching

To enable model caching, the application must specify the folder where to store cached blobs. It can be done like this

<pre><code>  ie = IECore()
  ie.set_config( {'CACHE_DIR' : 'path_to_cache'} )
  net = ie.read_network('sample.xml')
  ie.load_network(network=net, device_name='GPU', config=cache_config)
</pre></code>

With this code, if a device supports the Import/Export network capability, a cached blob is automatically created inside the path_to_cache directory `CACHE_DIR` config is set to the Core object. If device does not support Import/Export capability, the cache is not created and no error is thrown.

Depending on your device, the total time for loading network on application startup can be significantly reduced. Please also note that very first [IECore.load_network](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#ac9a2e043d14ccfa9c6bbf626cfd69fcc) call (before the cache is created) takes a slightly longer time to ‘export’ the compiled blob to a cache file.

![Model Caching](https://docs.openvinotoolkit.org/latest/caching_enabled.png)


## Even Faster: Use IECore.load_network('path_to_model')

In some cases, applications do not need to customize inputs and outputs every time. These applications always call `IECore.read_network()`, then `IECore.load_network(model='sample.xml')` and may be further optimized. For such cases, it's more convenient to load the network in a single call, as introduced in the 2021.4 release.

<pre><code>  ie = IECore()
  cfg = ie.set_config( {'CACHE_DIR' : 'path_to_cache_directory'} )
  ie.load_network(model=mod, device_name=dev, config=cfg)
</pre></code>

With model caching enabled, total load time is even faster - when `read_network()` is optimized as well

<pre><code>  ie = IECore()
  cfg = ie.set_config( {'CACHE_DIR' : 'path_to_cache_directory'} )

  ie.load_network(model=mod, device_name==dev, config=cfg)
</pre></code>

![Caching Times](https://docs.openvinotoolkit.org/latest/caching_times.png)

## Advanced Examples

Not every device supports network import/export capability. For those that don't, enabling caching has no effect. To check in advance if a particular device supports model caching, your application can use the following code:

<pre><code>  all_metrics = ie.get_metric(device_name=target_device, metric_name='SUPPORTED_METRICS')
  # Find the 'IMPORT_EXPORT_SUPPORT' metric in supported metrics
  allows_caching = all_metrics('IMPORT_EXPORT_SUPPORT')
</pre></code>
