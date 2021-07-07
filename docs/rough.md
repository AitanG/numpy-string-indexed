# Docs

## class friendly_matrix.ndarray

### Description

A structure for matrix-like data, which stores the data as a classic NumPy ndarray, and provides the option to reference by human-readable values.

The `fm.ndarray class`, plus the other functions exposed in the friendly_matrix package, are designed as substitutes for the NumPy `ndarray`, with comparable performance benchmarks and familiar, NumPy-style usage patterns.

Labels do not need to be specified for every dimension and index. There are four ways to initialize a `friendly_matrix.ndarray` instance:


### Args

* `array` (`numpy.ndarray`): NumPy array to wrap
* `dim_names=None (list)`: label for each dimension
* `*args_dim_arrays`: index labels for each dimension, or single dict mapping each dimension label to its corresponding index labels
* `**kwargs_dim_arrays`: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)


### `friendly_matrix.ndarray` Members

#### `friendly_matrix.ndarray.ndim` (int):

Number of dimensions

#### `friendly_matrix.ndarray.shape` (tuple):

Length of each dimension

#### `friendly_matrix.ndarray.size` (int):

Total number of elements in the array

#### `friendly_matrix.ndarray.dtype` (type):

Data type of the array

#### `friendly_matrix.ndarray.itemsize` (int):

Length of one element of the array in bytes

#### `friendly_matrix.ndarray.dim_length(dim)` (int):

**Arguments:**

* `dim`: dimension label or index

**Returns:** length of that dimension

#### `friendly_matrix.ndarray.take(*args, **kwargs)` (friendly_matrix.ndarray)

**Description:**

Takes a slice of the array according to the specified labels.

**Arguments:**

* `*args`: index labels to select for each dimension, or single dict mapping each dimension label to its corresponding index labels
* `**kwargs`: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

If no labels are specified for a dimension, the entire dimension is selected. If a single label is specified for a dimension, that dimension is dropped in the result.

A take operation can also be performed by calling a `friendly_matrix.ndarray` instance directly.

Example usage:

Casting:

```python
friendly_matrix.ndarray(array)
```

Dim arrays as** Arguments:**

```python
friendly_matrix.ndarray(
	array,
	['a', 'b'],
	['x', 'y', 'z'])
```

Dim arrays as dict:

```python
dim_arrays = {
	'a': ['x', 'y', 'z']
}
friendly_matrix.ndarray(
	array,
	['a', 'b'],
	dim_arrays)
```

Dim arrays as** kwArguments:**

```python
friendly_matrix.ndarray(
	array,
	['a', 'b'],
	a=['x', 'y', 'z'])
```

**Returns:** A new `friendly_matrix.ndarray` instance containing the filtered array

#### `friendly_matrix.ndarray.take_A(*args, **kwargs)` (numpy.ndarray)

Same as `take()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.get(*args, **kwargs)` (object)

**Description:**

Gets the single element by its labels.

**Arguments:**

* `*args`: index labels to select for each dimension, or single dict mapping each dimension label to its corresponding index labels
* `**kwargs`: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

A get operation can also be performed by calling a `friendly_matrix.ndarray` directly.

**Returns:** The element

#### `friendly_matrix.ndarray.set(val, *args, **kwargs)` (None)

**Description:**

Sets the single element by its labels.

**Arguments:**

* `val`: the updated value
* `*args`: index labels to select for each dimension, or single dict mapping each dimension label to its corresponding index labels
* `**kwargs`: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

#### `friendly_matrix.ndarray.copy()` (friendly_matrix.ndarray)

**Description:**

Creates a deep copy of the current object.

#### `friendly_matrix.ndarray.formatted(topological_order=None, formatter=None, display_dim_names=True)` (str)

**Description:**

Formats the `friendly_matrix.ndarray` instance as a nested list. All elements in the array are listed linearly under their dim index labels. The order in which dimensions are traversed can be set, as well as whether dim names are displayed alongside dim index labels, and how elements should be formatted before being appended to the result.

This is useful for displaying the labels and values of smaller matrices or slice results.

**Arguments:**

* `topological_order`: iterable representing the order in which dimensions should be traversed for output
* `formatter`: callable that formats an element for output
* `display_dim_names`: whether to display dim names alongside dim array labels

Example usage:

```python
prices.formatted(topological_order=["Year", "Size"],
				 formatter=price_formatter,
				 display_dim_names=True)
```

Output:

	Year = 2010:
		Size = small:
			$1.99
		Size = large:
			$2.99
	Year = 2020:
		Size = small:
			$2.99
		Size = large:
			$3.99


#### `friendly_matrix.ndarray.moveaxis(dim, new_dim)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style moveaxis operation on the `friendly_matrix.ndarray` instance. The ordering of dimensions is changed by moving one dimension to the position of another dimension.

**Arguments:**

* `dim`: the dimension to move
* `new_dim`: the dimension whose place `dim` will take

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.moveaxis_A(dim, new_dim)` (numpy.ndarray)

Same as `moveaxis()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.swapaxes(dim1, dim2)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style swapaxes operation on the `friendly_matrix.ndarray` instance. The ordering of dimensions is changed by swapping the positions of two dimensions.

**Arguments:**

* `dim1`: dimension
* `dim2`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.swapaxes_A(dim1, dim2)` (numpy.ndarray)

Same  as `swapaxis()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.transpose()` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style transpose operation on the `friendly_matrix.ndarray` instance. The ordering of the first two dimensions are swapped.

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.transpose_A()` (numpy.ndarray)

Same as `transpose()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.T` (friendly_matrix.ndarray)

Same as `transpose()`.

#### `friendly_matrix.ndarray.T_A` (numpy.ndarray)

Same as `transpose_A()`.

#### `friendly_matrix.ndarray.mean(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style mean computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the mean(s) along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.mean_A(axis)` (numpy.ndarray)

Same as `mean()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.std(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style std computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the standard deviation(s) along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.std_A(axis)` (numpy.ndarray)

Same as `std()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.var(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style var computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the variance(s) along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.var_A(axis)` (numpy.ndarray)

Same as `var()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.sum(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style sum computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the sum(s) along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.sum_A(axis)` (numpy.ndarray)

Same as `sum()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.prod(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style prod computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the product(s) along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.prod_A(axis)` (numpy.ndarray)

Same as `prod()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.min(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style min computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating minimum value(s) along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.min_A(axis)` (numpy.ndarray)

Same as `min()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.argmin(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style argmin computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the index or indices of the minimum value along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.argmin_A(axis)` (numpy.ndarray)

Same as `argmin()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.all(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style all computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating whether all the values along that dimension are truthy.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.all_A(axis)` (numpy.ndarray)

Same as `all()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.any(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style any computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the any of the values along that dimension are truthy.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.any_A(axis)` (numpy.ndarray)

Same as `any()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.cumsum(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style cumsum computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the cumulative sum along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance (with the same shape as the original)

#### `friendly_matrix.ndarray.cumsum_A(axis)` (numpy.ndarray)

Same as `cumsum()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.cumprod(axis)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style cumprod computation on the `friendly_matrix.ndarray` instance. Aggregates over a given dimension by calculating the cumulative product along that dimension.

**Arguments:**

* `axis`: dimension

**Returns:** The new `friendly_matrix.ndarray` instance (with the same shape as the original)

#### `friendly_matrix.ndarray.cumprod_A(axis)` (numpy.ndarray)

Same as `cumprod()`, except returns only the NumPy array.

#### `friendly_matrix.ndarray.squeeze()` (friendly_matrix.ndarray)

**Description:**

Removes any length 1 dimensions in the `friendly_matrix.ndarray` instance by aggregating over them.

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.ndarray.squeeze_A()` (numpy.ndarray)

Same as `squeeze()`, except returns only the NumPy array.


### Formatting

#### `friendly_matrix.formatted(friendly, topological_order=None, formatter=None, display_dim_names=True)` (str)

Equivalent to `friendly.formatted(topological_order, formatter, display_dim_names)`.

#### `friendly_matrix.from_formatted(formatted_friendly, dtype=numpy.str)` (friendly_matrix.ndarray)

Deserializes a string representation of a `friendly_matrix.ndarray` instance back into a new `friendly_matrix.ndarray` instance.

**Arguments:**

* `formatted_friendly`: the formatted `friendly_matrix.ndarray` instance
* `dtype`: the data type of the result `friendly_matrix.ndarray`

Assumes a valid string is provided.

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.from_formatted_A(formatted_friendly, dtype=numpy.str)` (friendly_matrix.ndarray)

Same as `from_formatted()`, except returns only the NumPy array.


### Computation

#### `friendly_matrix.compute_ndarray(dim_names, *args, dtype=numpy.float32)` (friendly_matrix.ndarray)

Generates a `friendly_matrix.ndarray` object by computing it and having it indexable the same way it's computed: using embedded loops over human-readable lists of values.

A friendly matrix is an ideal structure for storing and retrieving the results of computations over multiple variables. The `compute_ndarray()` function executes computations over all values of the input arrays and stores them in a new `friendly_matrix.ndarray` instance in a single step.

**Arguments:**

* `dim_names`: the name of each dimension
* `*args`: iterables or callables specifying how to calculate results
* `dtype`: the data type of the computed results

The `args` arguments should contain iterables or callables, which constitute a complete set of instructions for computing the result. The first argument must be an iterable, and the last argument must be a callable. A group of one or more consecutive iterable arguments are iterated over via their Cartesian product. The next argument, which is a callable, takes the values from the current iteration as arguments to run some user-defined code, which can optionally yield precomputations for use in subsequent callables further up the stack.

For intermediate callables should assemble precomputations in a dictionary, which is returned, to make them available to subsequent callables. For subsequent callables to access these precomputations, these callables should accept them as keyword arguments.

The final callable should return a value, which gets stored at a location in the `ndarray` specified by the values from the current iteration.

The `dim_names` argument should match the order of `args`.

The dim index labels in the result are set as the values of each iterable provided in `*args`.

Below is a bare-bones example of how `compute_ndarray()` can be used:

```python
iterable_a = [1, 2, 3]
iterable_b = [40, 50, 60]
iterable_c = [.7, .8, .9]

def callable_1(val_a, val_b):
	precomputations = {
		'intermediate_sum': val_a + val_b
	}
	return precomputations

def callable_2(val_a, val_b, val_c, **precomputations):
	final_result = precomputations['intermediate_sum'] * val_c
	return final_result

result_friendly_matrix = friendly_matrix.compute_ndarray(
	['a', 'b', 'c'],
	iterable_a,
	iterable_b,
	callable_1,
	iterable_c,
	callable_2)  # shape is (3, 3, 3)
```

#### `friendly_matrix.compute_ndarray_A(dim_names, *args, dtype=numpy.float32)` (friendly_matrix.ndarray)

Same as `compute_ndarray()`, except returns only the NumPy array.


### Methods

#### `friendly_matrix.take(friendly, *args, **kwargs)` (friendly_matrix.ndarray)

Equivalent to `friendly.take(*args, **kwargs)`.

#### `friendly_matrix.take_A(friendly, *args, *kwargs)` (numpy.ndarray)

Equivalent to `friendly.take_A(*args, **kwargs)`.

#### `friendly_matrix.get(friendly, *args, *kwargs)` (friendly_matrix.ndarray)

Equivalent to `friendly.get(*args, **kwargs)`.

#### `friendly_matrix.set(friendly, *args, *kwargs)` (friendly_matrix.ndarray)

Equivalent to `friendly.set(*args, **kwargs)`.

#### `friendly_matrix.copy(friendly)` (friendly_matrix.ndarray)

Equivalent to `friendly.copy()`.

### Operations

#### `friendly_matrix.concatenate(friendlies, axis=0)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style concatenate operation on the `friendly_matrix.ndarray` instance. Concatenates the provided `friendly_matrix.ndarray` instances along the specified dimension.

**Arguments:**

* `friendlies`: `friendly_matrix.ndarray` instances
* `axis`: the dimension along which to concatenate `friendlies`

**Returns:** The new `friendly_matrix.ndarray` instance

#### `friendly_matrix.concatenate_A(friendlies, axis=0)` numpy.ndarray)

Same as `concatenate()`, except returns only the NumPy array.

#### `friendly_matrix.stack(friendlies, axis_name, axis_array, axis=0)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style stack operation on the  `friendly_matrix.ndarray` instances. Stacks the provided `friendly_matrix.ndarray` instances along a new dimension.

**Arguments:**

* `friendlies`: `friendly_matrix.ndarray` instances
* `axis_name`: label for the new dimension
* `axis_array`: index labels for the new dimension
* `axis`: the dimension where the new dimension will be inserted

The `axis_array` argument should have the same length as `friendlies`.

#### `friendly_matrix.stack_A(friendlies, axis_name=None, axis_array=None, axis=None)` (friendly_matrix.ndarray)

Same as `stack()`, except returns only the NumPy array.

#### `friendly_matrix.vstack(friendlies)` (friendly_matrix.ndarray)

Equivalent to `concatenate(friendlies, axis=0)`. Can't be performed on one-dimensional arrays.

#### `friendly_matrix.vstack_A(friendlies)` (numpy.ndarray)

Same as `vstack()`, except returns only the NumPy array.

#### `friendly_matrix.hstack(friendlies)` (friendly_matrix.ndarray)

Equivalent to `concatenate(friendlies, axis=1)`

#### `friendly_matrix.hstack_A(friendlies)` (numpy.ndarray)

Same as `hstack()`, except returns only the NumPy array.

#### `friendly_matrix.flip(friendly, axis=None)` (friendly_matrix.ndarray)

**Description:**

Performs a NumPy-style flip operation on the  `friendly_matrix.ndarray` instances. Reverses the order of elements along the provided dimension(s).

**Arguments:**

* `friendly`: `friendly_matrix.ndarray` instance
* `axis`: dimension(s) along which to flip elements

The default value for `axis` of `None` results in a flip along all dimensions.

#### `friendly_matrix.flip_A(friendly, axis=None)` (numpy.ndarray)

Same as `flip()`, except returns only the NumPy array.

#### `friendly_matrix.fliplr(friendly)` (friendly_matrix.ndarray)

Equivalent to `friendly_matrix.flip(friendly, axis=0)`.

#### `friendly_matrix.fliplr_A(friendly)` (numpy.ndarray)

Same as `fliplr()`, except returns only the NumPy array.

#### `friendly_matrix.flipud(friendly)` (friendly_matrix.ndarray)

Equivalent to `friendly_matrix.flip(friendly, axis=1)`.

#### `friendly_matrix.flipud_A(friendly)` (numpy.ndarray)

Same as `flipud()`, except returns only the NumPy array.

#### `friendly_matrix.moveaxis(friendly, dim, new_dim)` (friendly_matrix.ndarray)

Equivalent to `friendly.moveaxis(axis)`.

#### `friendly_matrix.moveaxis_A(friendly, dim, new_dim)` (numpy.ndarray)

Equivalent to `friendly.moveaxis_A(axis)`.

#### `friendly_matrix.swapaxes(friendly, dim1, dim2)` (friendly_matrix.ndarray)

Equivalent to `friendly.swapaxes(axis)`.

#### `friendly_matrix.swapaxes_A(friendly, dim1, dim2)` (numpy.ndarray)

Equivalent to `friendly.swapaxes_A(axis)`.

#### `friendly_matrix.transpose(friendly)` (friendly_matrix.ndarray)

Equivalent to `friendly.transpose(axis)`.

#### `friendly_matrix.transpose_A(friendly)` (numpy.ndarray)

Equivalent to `friendly.transpose_A(axis)`.

#### `friendly_matrix.ndarray.mean(axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.mean(axis)`.

#### `friendly_matrix.mean_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.mean_A(axis)`.

#### `friendly_matrix.std(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.std(axis)`.

#### `friendly_matrix.std_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.std_A(axis)`.

#### `friendly_matrix.var(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.var(axis)`.

#### `friendly_matrix.var_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.var_A(axis)`.

#### `friendly_matrix.sum(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.sum(axis)`.

#### `friendly_matrix.sum_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.sum_A(axis)`.

#### `friendly_matrix.prod(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.prod(axis)`.

#### `friendly_matrix.prod_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.prod_A(axis)`.

#### `friendly_matrix.min(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.min(axis)`.

#### `friendly_matrix.min_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.min_A(axis)`.

#### `friendly_matrix.argmin(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.argmin(axis)`.

#### `friendly_matrix.argmin_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.argmin_A(axis)`.

#### `friendly_matrix.all(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.all(axis)`.

#### `friendly_matrix.all_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.all_A(axis)`.

#### `friendly_matrix.any(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.any(axis)`.

#### `friendly_matrix.any_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.any_A(axis)`.

#### `friendly_matrix.cumsum(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.cumsum(axis)`.

#### `friendly_matrix.cumsum_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.cumsum_A(axis)`.

#### `friendly_matrix.cumprod(friendly, axis)` (friendly_matrix.ndarray)

Equivalent to `friendly.cumprod(axis)`.

#### `friendly_matrix.cumprod_A(friendly, axis)` (numpy.ndarray)

Equivalent to `friendly.cumprod_A(axis)`.

#### `friendly_matrix.squeeze(friendly)` (friendly_matrix.ndarray)

Equivalent to `friendly.squeeze()`.

#### `friendly_matrix.squeeze_A(friendly)` (numpy.ndarray)

Equivalent to `friendly.squeeze_A()`.
