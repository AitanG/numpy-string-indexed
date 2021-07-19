.. title:: Computing arrays


Computing arrays
================

.. py:function:: friendly_matrix.compute_ndarray(dim_names, *args[, dtype=numpy.float32]) -> friendly_matrix.ndarray

	Generates a ``friendly_matrix.ndarray`` object by computing it and having it indexable the same way it's computed: using embedded loops over human-readable lists of values.

	A friendly matrix is an ideal structure for storing and retrieving the results of computations over multiple variables. The `compute_ndarray()` function executes computations over all values of the input arrays and stores them in a new ``friendly_matrix.ndarray`` instance in a single step.

	:param dim_names: the name of each dimension
	:param \*args: iterables or callables specifying how to calculate results
	:param dtype: the data type of the computed results

	The `args` arguments should contain iterables or callables, which constitute a complete set of instructions for computing the result. The first argument must be an iterable, and the last argument must be a callable. A group of one or more consecutive iterable arguments are iterated over via their Cartesian product. The next argument, which is a callable, takes the values from the current iteration as arguments to run some user-defined code, which can optionally yield precomputations for use in subsequent callables further up the stack.

	Any intermediate callables should assemble precomputations in a dictionary, which is returned, in order to make them available to subsequent callables. For subsequent callables to access these precomputations, these callables should accept them as keyword arguments.

	The final callable should return a value, which gets stored at a location in the ``friendly_matrix.ndarray`` specified by the values from the current iteration.

	The ``dim_names`` argument should match the order of ``args``.

	The dim index labels in the result are set as the values of each iterable provided in ``args``.

	Below is a bare-bones example of how ``compute_ndarray()`` can be used::

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

.. py:function:: friendly_matrix.compute_ndarray_A(dim_names, *args[, dtype=numpy.float32]) -> friendly_matrix.ndarray

	Same as :py:func:`compute_ndarray()`, except returns only the NumPy array.