NumPy String-Indexed
===

NumPy String-Indexed is a NumPy extension that allows arrays to be indexed using descriptive string labels, rather than conventional zero-indexing. When a friendly matrix object is initialized, labels are assigned to each array index and each dimension, and they stick to the array after NumPy-style operations such as transposing, concatenating, and aggregating. This prevents Python programmers from having to keep track mentally of what each axis and each index represents, instead making each reference to the array in code naturally self-documenting.

NumPy String-Indexed is especially useful for applications like machine learning, scientific computing, and data science, where there is heavy use of multidimensional arrays.

The friendly matrix object is implemented as a lightweight wrapper around a NumPy ndarray. It's easy to add to a new or existing project to make it easier to maintain code, and has a negligible memory and performance overhead compared to the size of array (![equation](http://www.sciweavers.org/tex2img.php?eq=O%28x+%2B+y+%2B+z%29+%3C+O%28x+%5Ctimes+y+%5Ctimes+z%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)).


## Basic functionality

It's recommended to import NumPy String-Indexed idiomatically as `fm`:

```python
import friendly_matrix as fm
```

Labels are provided during object construction and can optionally be used in place of numerical indices for slicing and indexing.

The example below shows how to construct a friendly matrix containing an image with three color channels:

```python
image = fm.ndarray(
	numpy_ndarray_image,  # np.ndarray with shape (3, 100, 100)
	dim_names=['color_channel', 'top_to_bottom', 'left_to_right'],
	color_channel=['R', 'G', 'B'])
```

The matrix can then be sliced this way:

```python
# friendly matrix with shape (100, 100)
r_channel = image(color_channel='R')

# an integer
g_top_left_pixel_value = image('G', 0, 0)

# friendly matrix with shape (100, 50)
br_channel_left_half = image(
	color_channel=('B', 'R'),
	left_to_right=range(image.dim_length('left_to_right') // 2))

```


## Matrix operations

Friendly matrix objects can be operated on just like NumPy ndarrays with minimal overhead. The package contains separate implementations of most of the relevant NumPy ndarray operations, taking advantage of labels. For example:

```python
side_by_side = fm.concatenate((image1, image2), axis='left_to_right')
```

An optimized alternative is to perform label-less operations, by adding `"_A"` (for "array") to the operation name:

```python
side_by_side_arr = fm.concatenate_A((image1, image2), axis='left_to_right')
```

If it becomes critical to optimize within a particular scope, it's recommended to shed labels before operating:

```python
for image in huge_list:
	image_processor(image.A)
```


## Computing matrices

A friendly matrix is an ideal structure for storing and retrieving the results of computations over multiple variables. The `fm.compute_ndarray()` function executes computations over all values of the input arrays and stores them in a new `fm.ndarray` in a single step:

```python
'''Collect samples from a variety of normal distributions'''

import numpy as np

n_samples_list = [1, 10, 100, 1000]
mean_list = list(range(-21, 21))
var_list = [1E1, 1E0, 1E-1, 1E-2, 1E-3]

results = fm.compute_ndarray(
	['# Samples', 'Mean', 'Variance']
	n_samples_list,
	mean_list,
	var_list,
	normal_sampling_function,
	dtype=np.float32)

# friendly matrices can be sliced using dicts
print(results({
	'# Samples': 100,
	'Mean': 0,
	'Variance': 1,
}))
```


## Formatting matrices

The `fm.formatted()` function displays a friendly matrix as a nested list. This is useful for displaying the labels and values of smaller matrices, or slicing results:

```python
mean_0_results = results({
	'# Samples': (1, 1000),
	'Mean': 0,
	'Variance': (10, 1, 0.1),
})
formatted = fm.formatted(
	mean_0_results,
	formatter=lambda n: round(n, 1))

print(formatted)

'''
Example output:

# Samples = 1:
	Variance = 10:
		2.2
	Variance = 1:
		-0.9
	Variance = 0.1:
		0.1
# Samples = 1000:
	Variance = 10:
		-0.2
	Variance = 1:
		-0.0
	Variance = 0.1:
		0.0
'''
```

## Documentation

Complete documentation is forthcoming.