.. title:: Array initialization and slicing


Array initialization and slicing
================================

``friendly_matrix.ndarray`` methods
-----------------------------------

.. py:class:: friendly_matrix.ndarray(array[, dim_names=None[, *args_dim_arrays[, **kwargs_dim_arrays]]])

	A structure for matrix-like data, which stores the data as a classic NumPy ``ndarray``, and provides the option to reference by human-readable values.

	This class, plus the other functions exposed in the friendly_matrix package, are designed as substitutes for the NumPy ``ndarray``, with comparable performance benchmarks and familiar, NumPy-style usage patterns.

	Labels do not need to be specified for every dimension and index. There are four ways to initialize a ``friendly_matrix.ndarray`` instance:

	Casting::

		friendly_matrix.ndarray(array)

	Dimension arrays as arguments::

		friendly_matrix.ndarray(
			array,
			['a', 'b'],
			['x', 'y', 'z'])

	Dimension arrays as dict::

		dim_arrays = {
			'a': ['x', 'y', 'z']
		}
		friendly_matrix.ndarray(
			array,
			['a', 'b'],
			dim_arrays)

	Dimension arrays as keyword arguments::

		friendly_matrix.ndarray(
			array,
			['a', 'b'],
			a=['x', 'y', 'z'])

	:param array: NumPy array to wrap
	:param dim_names: label for each dimension
	:param \*args_dim_arrays: index labels for each dimension, or single dict mapping each dimension label to its corresponding index labels
	:param \*\*kwargs_dim_arrays: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

	.. py:property:: friendly_matrix.ndarray.ndim

		:type: int

		Number of dimensions

	.. py:property:: friendly_matrix.ndarray.shape

		:type: tuple

		Length of each dimension

	.. py:property:: friendly_matrix.ndarray.size

		:type: int

		Total number of elements in the array

	.. py:property:: friendly_matrix.ndarray.dtype

		:type: type

		Data type of the array

	.. py:property:: friendly_matrix.ndarray.itemsize

		:type: int

		Length of one element of the array in bytes

	.. py:method:: friendly_matrix.ndarray.dim_length(dim) -> int

		:param dim: dimension label or index

		:return: length of that dimension

	.. py:method:: friendly_matrix.ndarray.take(*args, **kwargs) -> friendly_matrix.ndarray

		Takes a slice of the array according to the specified labels.

		:param \*args: index labels to select for each dimension, or single dict mapping each dimension label to its corresponding index labels
		:param \*\*kwargs: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

		If no labels are specified for a dimension, the entire dimension is selected. If a single label is specified for a dimension, that dimension is dropped in the result.

		A take operation can also be performed by calling a ``friendly_matrix.ndarray`` instance directly.

		Example usage:

		TODO

		:return: A new ``friendly_matrix.ndarray`` instance containing the filtered array

	.. py:method:: friendly_matrix.ndarray.take_A(*args, **kwargs) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.take()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.get(*args, **kwargs) -> object

		Gets the single element by its labels.

		:param \*args: index labels to select for each dimension, or single dict mapping each dimension label to its corresponding index labels
		:param \*\*kwargs: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

		A get operation can also be performed by calling a ``friendly_matrix.ndarray`` directly.

		:return: The element

	.. py:method:: friendly_matrix.ndarray.set(val, *args, **kwargs) -> None

		Sets the single element by its labels.

		:param val: the updated value
		:param \*args: index labels to select for each dimension, or single dict mapping each dimension label to its corresponding index labels
		:param \*\*kwargs: index labels for each dimension (only if specified dimensions are argument- and keyword-friendly)

	.. py:method:: friendly_matrix.ndarray.copy() -> friendly_matrix.ndarray

		Creates a deep copy of the current object.


Module functions
----------------
.. py:function:: friendly_matrix.take(friendly, *args, **kwargs) -> friendly_matrix.ndarray

	Equivalent to ``friendly.take(*args, **kwargs)``.

	See :py:meth:`friendly_matrix.ndarray.take()`.

.. py:function:: friendly_matrix.take_A(friendly, *args, *kwargs) -> numpy.ndarray

	Equivalent to ``friendly.take_A(*args, **kwargs)``.

	See :py:meth:`friendly_matrix.ndarray.take_A()`.

.. py:function:: friendly_matrix.get(friendly, *args, *kwargs) -> friendly_matrix.ndarray

	Equivalent to ``friendly.get(*args, **kwargs)``.

	See :py:meth:`friendly_matrix.ndarray.get()`.

.. py:function:: friendly_matrix.set(friendly, *args, *kwargs) -> friendly_matrix.ndarray

	Equivalent to ``friendly.set(*args, **kwargs)``.

	See :py:meth:`friendly_matrix.ndarray.set()`.

.. py:function:: friendly_matrix.copy(friendly) -> friendly_matrix.ndarray

	Equivalent to ``friendly.copy()``.

	See :py:meth:`friendly_matrix.ndarray.copy()`.