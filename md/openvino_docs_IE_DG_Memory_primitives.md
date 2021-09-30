# Inference Engine Memory primitives

## Blobs

InferenceEngine::Blob is the main class intended for working with memory. Using this class you can read and write memory, get information about the memory structure etc.

The right way to create Blob objects with a specific layout is to use constructors with [ie_api.TensorDesc](https://docs.openvinotoolkit.org/2021.1/ie_python_api/classie__api_1_1TensorDesc.html).

<pre><code>
  tensor_desc = TensorDesc(precision="FP32", dims=(1, 3, 227, 227), layout='NCHW')
  input_blob = Blob(tensor_description, some_input_data)
</code></pre>

## Layouts

InferenceEngine::TensorDesc is a special class that provides layout format description.

This class allows to create planar layouts using the standard formats (like InferenceEngine::Layout::NCDHW, InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NC, InferenceEngine::Layout::C and etc) and also non-planar layouts using InferenceEngine::BlockingDesc.

In order to create a complex layout you should use InferenceEngine::BlockingDesc which allows to define the blocked memory with offsets and strides.

<pre><code>C++

</code></pre>

## Examples

1. You can define a blob with dimensions {N: 1, C: 25, H: 20, W: 20} and format NHWC with using next parameters:
<pre><code>C++
  InferenceEngine::BlockingDesc({1, 20, 20, 25}, {0, 2, 3, 1}); // or
  InferenceEngine::BlockingDesc({1, 20, 20, 25}, InferenceEngine::Layout::NHWC);
</code></pre>


2. If you have a memory with real dimensions {N: 1, C: 25, H: 20, W: 20} but with channels which are blocked by 8, you can define it using next parameters:
<pre><code>
  C++
  InferenceEngine::BlockingDesc({1, 4, 20, 20, 8}, {0, 1, 2, 3, 1})
</code></pre>

3. Also you can set strides and offsets if layout contains it.
4. If you have a complex blob layout and you don’t want to calculate the real offset to data you can use methods InferenceEngine::TensorDesc::offset(size_t l) or InferenceEngine::TensorDesc::offset(SizeVector v).

For example:
<pre><code>
  C++
  InferenceEngine::BlockingDesc blk({1, 4, 20, 20, 8}, {0, 1, 2, 3, 1});
  InferenceEngine::TensorDesc tdesc(FP32, {1, 25, 20, 20}, blk);
  tdesc.offset(0); // = 0
  tdesc.offset(1); // = 8
  tdesc.offset({0, 0, 0, 2}); // = 16
  tdesc.offset({0, 1, 0, 2}); // = 17
</code></pre>

5. If you would like to create a TensorDesc with a planar format and for N dimensions (N can be different 1, 2, 4 and etc), you can use the method InferenceEngine::TensorDesc::getLayoutByDims.

<pre><code>
  C++
  InferenceEngine::TensorDesc::getLayoutByDims({1}); // InferenceEngine::Layout::C
  InferenceEngine::TensorDesc::getLayoutByDims({1, 2}); // InferenceEngine::Layout::NC
  InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3, 4}); // InferenceEngine::Layout::NCHW
  InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3}); // InferenceEngine::Layout::BLOCKED
  InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3, 4, 5}); // InferenceEngine::Layout::NCDHW
  InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3, 4, 5, ...}); // InferenceEngine::Layout::BLOCKED
</code></pre>

