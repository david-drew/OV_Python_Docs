<style>
  r { color: Red }
  pink { color: Pink }
  o { color: Orange }
  g { color: Green }
</style>

# Bfloat16 Inference

# Disclaimer
Inference Engine with the bfloat16 inference implemented on CPU must support the native *avx512_bf16* instruction and therefore the bfloat16 data format. It is possible to use bfloat16 inference in simulation mode on platforms with Intel® Advanced Vector Extensions 512 (Intel® AVX-512), but it leads to significant performance degradation in comparison with FP32 or native avx512_bf16 instruction usage.

# Introduction
Bfloat16 computations (referred to as BF16) is the Brain Floating-Point format with 16 bits. This is a truncated 16-bit version of the 32-bit IEEE 754 single-precision floating-point format FP32. BF16 preserves 8 exponent bits as FP32 but reduces precision of the sign and mantissa from 24 bits to 8 bits.

[IMG](https://docs.openvinotoolkit.org/latest/bf16_format.png)

Preserving the exponent bits keeps BF16 to the same range as the FP32 (~1e-38 to ~3e38). This simplifies conversion between two data types: you just need to skip or flush to zero 16 low bits. Truncated mantissa leads to occasionally less precision, but according to investigations, neural networks are more sensitive to the size of the exponent than the mantissa size. Also, in lots of models, precision is needed close to zero but not so much at the maximum range. Another useful feature of BF16 is possibility to encode INT8 in BF16 without loss of accuracy, because INT8 range completely fits in BF16 mantissa field. It reduces data flow in conversion from INT8 input image data to BF16 directly without intermediate representation in FP32, or in combination of INT8 inference and BF16 layers.

See the [BFLOAT16 – Hardware Numerics Definition" white paper](https://software.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf) for more bfloat16 format details.

There are two ways to check if CPU device can support bfloat16 computations for models:

1. Query the instruction set via system lscpu | grep avx512_bf16 or cat /proc/cpuinfo | grep avx512_bf16.
2. Use Query API with METRIC_KEY(OPTIMIZATION_CAPABILITIES), which should return BF16 in the list of CPU optimization options:

*DAVID*:  Find Python Equivalent for METRIC{"OPTIMIZATION_CAPABILITIES"}

<pre><code>
  ie = IECore()
  net = ie.read_network("sample.xml")
  exec_net = ie.load_network(network=net, device_name="CPU")
  cpu_caps = exec_net.get_metric("OPTIMIZATION_CAPABILITIES")
</code></pre>

The current Inference Engine solution for bfloat16 inference uses Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) and supports inference of the significant number of layers in BF16 computation mode.

## Lowering Inference Precision

Lowering precision to increase performance is widely used for optimization of inference. The bfloat16 data type usage on CPU for the first time opens the possibility of default optimization approach. The embodiment of this approach is to use the optimization capabilities of the current platform to achieve maximum performance while maintaining the accuracy of calculations within the acceptable range.

Using Bfloat16 precision provides the following performance benefits:

1. Faster multiplication of two BF16 numbers because of shorter mantissa of bfloat16 data.
2. No need to support denormals and handling exceptions as this is a performance optimization.
3. Fast conversion of float32 to bfloat16 and vice versa.
4. Reduced size of data in memory, as a result, larger models fit in the same memory bounds.
5. Reduced amount of data that must be transferred, as a result, reduced data transition time

For default optimization on CPU, the source model is converted from FP32 or FP16 to BF16 and executed internally on platforms with native BF16 support. In this case, KEY_ENFORCE_BF16 is set to YES. The code below demonstrates how to check if the key is set:

<pre><code>
  ie = IECore()
  net = ie.read_network("sample.xml")
  exec_net = ie.load_network(network=net, device_name="CPU")
  cpu_caps = exec_net.get_config("KEY_ENFORCE_BF16")
</code></pre>

To disable BF16 internal transformations, set the KEY_ENFORCE_BF16 to NO. In this case, the model infers as is without modifications with precisions that were set on each layer edge.

<pre><code>
  bf16_config = {"KEY_ENFORCE_BF16" : "YES"}

  ie = IECore()
  net = ie.read_network("sample.xml")
  exec_net = ie.load_network(network=net, device_name="CPU", config = bf16_config)
</code></pre>
  
An exception with the message <pink>Platform doesn't support BF16 format</pink> is formed in case of setting KEY_ENFORCE_BF16 to YES on CPU without native BF16 support or BF16 simulation mode.

Low-Precision 8-bit integer models cannot be converted to BF16, even if bfloat16 optimization is set by default.
  
</code></pre>











