.. title:: Array initialization and slicing


Array initialization and slicing
================================

``friendly_matrix.ndarray`` methods
-----------------------------------

.. py:class:: friendly_matrix.ndarray(array[, dim_names=None[, *args_dim_arrays[, **kwargs_dim_arrays]]])

	A structure for matrix-like data, which stores the data as a classic NumPy ``ndarray``, and provides the option to reference by human-readable values.

	This class, plus the other functions exposed in the friendly_matrix package, are designed as substitutes for the NumPy ``ndarray``, with comparable performance benchmarks and familiar, NumPy-style usage patterns.

	Labels do not need to be specified for every dimension and index. There are four ways to initialize a ``friendly_matrix.ndarray`` instance using the constructor, all of which involve assigning new labels to an existing NumPy ``ndarray``. The other main way to create a new ``friendly_matrix.ndarray`` is by calling :py:func:`friendly_matrix.compute_ndarray()`. The four ways are demonstrated below. In the examples, we assume the array ``array`` consists of two dimensions, for *size* and *n_passengers*. Dimension *size* has length 3, for *small*, *medium*, and *large*, and dimension *n_passengers* goes from 0 to 4.

	**1. Casting**::

		rockets = friendly_matrix.ndarray(array)

	Note: This creates an unlabeled ``friendly_matrix.ndarray`` instance.

	**2. Dimension arrays as arguments**::

		rockets = friendly_matrix.ndarray(
			array,
			['size', 'n_passengers'],
			['small', 'medium', 'large'])

	**3. Dimension arrays as dict**::

		dim_arrays = {
			'size': ['small', 'medium', 'large']
		}
		rockets = friendly_matrix.ndarray(
			array,
			['size', 'n_passengers'],
			dim_arrays)

	**4. Dimension arrays as keyword arguments**::

		rockets = friendly_matrix.ndarray(
			array,
			['size', 'n_passengers'],
			size=['small', 'medium', 'large'])

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

		If no labels are specified for a dimension, the entire dimension is selected. If a single label not wrapped in a list is specified for a dimension, that dimension is dropped in the result.

		A take operation can also be performed by calling a ``friendly_matrix.ndarray`` instance directly. It's recommended to use this shorthand for style.

		The three ways of using ``take()`` are demonstrated below. In the examples, we assume the array ``rockets`` consists of two dimensions, *size* and *n_passengers*. Dimension *size* has indices named *small*, *medium*, and *large*, and dimension *n_passengers* goes from 0 to 4.

		**1. Dimension arrays as arguments**::

			rockets('large', [2, 3])

		**2. Dimension arrays as dict**::

			rockets({
				'size': 'large',
				'n_passengers': [2, 3]
			})

		**3. Dimension arrays as keyword arguments**::

			rockets(size='large', n_passengers=[2, 3])

		**Note:** In the above examples, the shape of the result is ``(2,)``, because passing in the single value ``'large'`` for the first dimension causes the dimension to be dropped from the result. Passing in ``['large']`` instead would result in a shape of ``(1, 2)``.

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