.. title:: Array operations


Array operations
================

``friendly_matrix.ndarray`` methods
-----------------------------------

.. py:class:: friendly_matrix.ndarray
	:noindex:

	.. py:method:: friendly_matrix.ndarray.moveaxis(dim, new_dim) -> friendly_matrix.ndarray

		Performs a NumPy-style moveaxis operation on the ``friendly_matrix.ndarray`` instance. The ordering of dimensions is changed by moving one dimension to the position of another dimension.

		:param dim: the dimension to move
		:param new_dim: the dimension whose place `dim` will take

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.moveaxis_A(dim, new_dim) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.moveaxis()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.swapaxes(dim1, dim2) -> friendly_matrix.ndarray

		Performs a NumPy-style swapaxes operation on the ``friendly_matrix.ndarray`` instance. The ordering of dimensions is changed by swapping the positions of two dimensions.

		:param dim1: dimension
		:param dim2: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.swapaxes_A(dim1, dim2) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.swapaxis()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.transpose() -> friendly_matrix.ndarray

		Performs a NumPy-style transpose operation on the ``friendly_matrix.ndarray`` instance. The ordering of the first two dimensions are swapped.

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.transpose_A() -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.transpose()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.T -> friendly_matrix.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.transpose()`.

	.. py:method:: friendly_matrix.ndarray.T_A -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.transpose_A()`.

	.. py:method:: friendly_matrix.ndarray.mean(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style mean computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the mean(s) along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.mean_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.mean()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.std(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style std computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the standard deviation(s) along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.std_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.std()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.var(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style var computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the variance(s) along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.var_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.var()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.sum(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style sum computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the sum(s) along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.sum_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.sum()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.prod(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style prod computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the product(s) along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.prod_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.prod()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.min(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style min computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating minimum value(s) along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.min_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.min()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.argmin(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style argmin computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the index or indices of the minimum value along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.argmin_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.argmin()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.all(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style all computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating whether all the values along that dimension are truthy.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.all_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.all()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.any(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style any computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the any of the values along that dimension are truthy.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.any_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.any()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.cumsum(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style cumsum computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the cumulative sum along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance (with the same shape as the original)

	.. py:method:: friendly_matrix.ndarray.cumsum_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.cumsum()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.cumprod(axis) -> friendly_matrix.ndarray

		Performs a NumPy-style cumprod computation on the ``friendly_matrix.ndarray`` instance. Aggregates over a given dimension by calculating the cumulative product along that dimension.

		:param axis: dimension

		:return: The new ``friendly_matrix.ndarray`` instance (with the same shape as the original)

	.. py:method:: friendly_matrix.ndarray.cumprod_A(axis) -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.cumprod()`, except returns only the NumPy array.

	.. py:method:: friendly_matrix.ndarray.squeeze() -> friendly_matrix.ndarray

		Removes any length 1 dimensions in the ``friendly_matrix.ndarray`` instance by aggregating over them.

		:return: The new ``friendly_matrix.ndarray`` instance

	.. py:method:: friendly_matrix.ndarray.squeeze_A() -> numpy.ndarray

		Same as :py:meth:`friendly_matrix.ndarray.squeeze()`, except returns only the NumPy array.


Module functions
----------------
.. py:function:: friendly_matrix.concatenate(friendlies, axis=0) -> friendly_matrix.ndarray

	Performs a NumPy-style concatenate operation on the ``friendly_matrix.ndarray`` instance. Concatenates the provided ``friendly_matrix.ndarray`` instances along the specified dimension.

	:param friendlies: ``friendly_matrix.ndarray`` instances
	:param axis: the dimension along which to concatenate `friendlies`

	:return: The new ``friendly_matrix.ndarray`` instance

.. py:function:: friendly_matrix.concatenate_A(friendlies, axis=0) numpy.ndarray)

	Same as :py:func:`friendly_matrix.concatenate()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.stack(friendlies, axis_name, axis_array, axis=0) -> friendly_matrix.ndarray

	Performs a NumPy-style stack operation on the  ``friendly_matrix.ndarray`` instances. Stacks the provided ``friendly_matrix.ndarray`` instances along a new dimension.

	:param friendlies: ``friendly_matrix.ndarray`` instances
	:param axis_name: label for the new dimension
	:param axis_array: index labels for the new dimension
	:param axis: the dimension where the new dimension will be inserted

	The ``axis_array`` argument should have the same length as ``friendlies``.

.. py:function:: friendly_matrix.stack_A(friendlies, axis_name=None, axis_array=None, axis=None) -> friendly_matrix.ndarray

	Same as :py:func:`friendly_matrix.stack()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.vstack(friendlies) -> friendly_matrix.ndarray

	Equivalent to ``concatenate(friendlies, axis=0)``. Can't be performed on one-dimensional arrays`.

	See :py:func:`friendly_matrix.concatenate()`.

.. py:function:: friendly_matrix.vstack_A(friendlies) -> numpy.ndarray

	Same as :py:func:`friendly_matrix.vstack()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.hstack(friendlies) -> friendly_matrix.ndarray

	Equivalent to ``concatenate(friendlies, axis=1)``.

	See :py:func:`friendly_matrix.concatenate()`.

.. py:function:: friendly_matrix.hstack_A(friendlies) -> numpy.ndarray

	Same as :py:func:`friendly_matrix.hstack()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.flip(friendly, axis=None) -> friendly_matrix.ndarray

	Performs a NumPy-style flip operation on the  ``friendly_matrix.ndarray`` instances. Reverses the order of elements along the provided dimension(s).

	:param friendly: ``friendly_matrix.ndarray`` instance
	:param axis: dimension(s) along which to flip elements

	The default value for ``axis`` of ``None`` results in a flip along all dimensions.

.. py:function:: friendly_matrix.flip_A(friendly, axis=None) -> numpy.ndarray

	Same as :py:func:`friendly_matrix.flip()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.fliplr(friendly) -> friendly_matrix.ndarray

	Equivalent to ``friendly_matrix.flip(friendly, axis=0)``.

	See :py:func:`friendly_matrix.flip()`.

.. py:function:: friendly_matrix.fliplr_A(friendly) -> numpy.ndarray

	Same as :py:func:`friendly_matrix.fliplr()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.flipud(friendly) -> friendly_matrix.ndarray

	Equivalent to ``friendly_matrix.flip(friendly, axis=1)``.

	See :py:func:`friendly_matrix.flip()`.

.. py:function:: friendly_matrix.flipud_A(friendly) -> numpy.ndarray

	Same as :py:func:`friendly_matrix.flipud()`, except returns only the NumPy array.

.. py:function:: friendly_matrix.moveaxis(friendly, dim, new_dim) -> friendly_matrix.ndarray

	Equivalent to ``friendly.moveaxis(axis)``.

	See :py:meth:`friendly_matrix.ndarray.moveaxis()`.

.. py:function:: friendly_matrix.moveaxis_A(friendly, dim, new_dim) -> numpy.ndarray

	Equivalent to ``friendly.moveaxis_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.moveaxis_A()`.

.. py:function:: friendly_matrix.swapaxes(friendly, dim1, dim2) -> friendly_matrix.ndarray

	Equivalent to ``friendly.swapaxes(axis)``.

	See :py:meth:`friendly_matrix.ndarray.swapaxes()`.

.. py:function:: friendly_matrix.swapaxes_A(friendly, dim1, dim2) -> numpy.ndarray

	Equivalent to ``friendly.swapaxes_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.swapaxes_A()`.

.. py:function:: friendly_matrix.transpose(friendly) -> friendly_matrix.ndarray

	Equivalent to ``friendly.transpose(axis)``.

	See :py:meth:`friendly_matrix.ndarray.transpose()`.

.. py:function:: friendly_matrix.transpose_A(friendly) -> numpy.ndarray

	Equivalent to ``friendly.transpose_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.transpose_A()`.

.. py:function:: friendly_matrix.mean(axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.mean(axis)``.

	See :py:meth:`friendly_matrix.ndarray.mean()`.

.. py:function:: friendly_matrix.mean_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.mean_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.mean_A()`.

.. py:function:: friendly_matrix.std(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.std(axis)``.

	See :py:meth:`friendly_matrix.ndarray.std()`.

.. py:function:: friendly_matrix.std_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.std_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.std_A()`.

.. py:function:: friendly_matrix.var(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.var(axis)``.

	See :py:meth:`friendly_matrix.ndarray.var()`.

.. py:function:: friendly_matrix.var_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.var_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.var_A()`.

.. py:function:: friendly_matrix.sum(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.sum(axis)``.

	See :py:meth:`friendly_matrix.ndarray.sum()`.

.. py:function:: friendly_matrix.sum_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.sum_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.sum_A()`.

.. py:function:: friendly_matrix.prod(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.prod(axis)``.

	See :py:meth:`friendly_matrix.ndarray.prod()`.

.. py:function:: friendly_matrix.prod_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.prod_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.prod_A()`.

.. py:function:: friendly_matrix.min(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.min(axis)``.

	See :py:meth:`friendly_matrix.ndarray.min()`.

.. py:function:: friendly_matrix.min_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.min_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.min_A()`.

.. py:function:: friendly_matrix.argmin(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.argmin(axis)``.

	See :py:meth:`friendly_matrix.ndarray.argmin()`.

.. py:function:: friendly_matrix.argmin_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.argmin_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.argmin_A()`.

.. py:function:: friendly_matrix.all(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.all(axis)``.

	See :py:meth:`friendly_matrix.ndarray.all()`.

.. py:function:: friendly_matrix.all_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.all_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.all_A()`.

.. py:function:: friendly_matrix.any(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.any(axis)``.

	See :py:meth:`friendly_matrix.ndarray.any()`.

.. py:function:: friendly_matrix.any_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.any_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.any_A()`.

.. py:function:: friendly_matrix.cumsum(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.cumsum(axis)``.

	See :py:meth:`friendly_matrix.ndarray.cumsum()`.

.. py:function:: friendly_matrix.cumsum_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.cumsum_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.cumsum_A()`.

.. py:function:: friendly_matrix.cumprod(friendly, axis) -> friendly_matrix.ndarray

	Equivalent to ``friendly.cumprod(axis)``.

	See :py:meth:`friendly_matrix.ndarray.cumprod()`.

.. py:function:: friendly_matrix.cumprod_A(friendly, axis) -> numpy.ndarray

	Equivalent to ``friendly.cumprod_A(axis)``.

	See :py:meth:`friendly_matrix.ndarray.cumprod_A()`.

.. py:function:: friendly_matrix.squeeze(friendly) -> friendly_matrix.ndarray

	Equivalent to ``friendly.squeeze()``.

	See :py:meth:`friendly_matrix.ndarray.squeeze()`.

.. py:function:: friendly_matrix.squeeze_A(friendly) -> numpy.ndarray

	Equivalent to ``friendly.squeeze_A()``.

	See :py:meth:`friendly_matrix.ndarray.squeeze_A()`.