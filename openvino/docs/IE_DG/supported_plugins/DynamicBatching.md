# How to Use Dynamic Batching

Dynamic Batching is a feature that allows you to dynamically change batch size for inference calls within a preset batch size limit. This feature might be useful when batch size is unknown beforehand, and using extra large batch size is not desired or impossible due to resource limitations. For example, face detection with person age, gender, or mood recognition is a typical usage scenario.

You can activate Dynamic Batching by setting the "DYN_BATCH_ENABLED" flag to "YES" in a configuration map that is passed to the plugin while loading a network. This configuration creates an ExecutableNetwork object that will allow setting batch size dynamically in all of its infer requests using SetBatch() method. The batch size that was set in passed CNNNetwork object will be used as a maximum batch size limit.



@snippet snippets/DynamicBatching.py part0

@sphinxdirective
..tab:: C++
	..code-block:: c
	@snippet snippets/DynamicBatching.cpp part0
..tab:: Python
	..code-block:: python
	@snippet snippets/DynamicBatching.py part0
@endsphinxdirective

## Limitations

Currently, certain limitations for the use of Dynamic Batching exist:

* Use Dynamic Batching with CPU and GPU plugins only.
* Use Dynamic Batching on topologies that consist of certain layers only:
  * Convolution
  * Deconvolution
  * Activation
  * LRN
  * Pooling
  * FullyConnected
  * SoftMax
  * Split
  * Concatenation
  * Power
  * Eltwise
  * Crop
  * BatchNormalization
  * Copy

Do NOT use:
* layers that might arbitrary change tensor shape (such as Flatten, Permute, Reshape)
* layers specific to object detection topologies (ROIPooling, ProirBox, DetectionOutput)
* custom layers. 
 
Topology analysis is performed during the process of loading a network into plugin, and if the topology is not supported, an exception is generated.

