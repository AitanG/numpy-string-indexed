.. title:: User's guide


User's guide
============


Below is an overview of the extensions NumPy String-Indexed offers. Functionality can be categorized into: array operations, computing arrays, and formatting arrays.

The examples below build on those from Index.


Array operations
----------------

Friendly matrix objects can be operated on just like NumPy ``ndarray`` s with minimal overhead. The package contains separate implementations of most of the relevant NumPy ``ndarray`` operations, taking advantage of labels. For example::


	side_by_side = fm.concatenate((image1, image2), axis='left_to_right')


An optimized alternative is to perform label-less operations, by adding ``"_A"`` (for "array") to the operation name::


	side_by_side_arr = fm.concatenate_A((image1, image2), axis='left_to_right')


If it becomes important to optimize within a particular scope, it's recommended to shed labels before operating::


	for image in huge_list:
		image_processor(image.A)



Computing arrays
----------------

A friendly matrix is an ideal structure for storing and retrieving the results of computations over multiple variables. The ``compute_ndarray()`` function executes computations over all values of the input arrays and stores them in a new Friendly Matrix ``ndarray`` instance in a single step::


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



Formatting arrays
-----------------

The ``formatted()`` function displays a friendly matrix as a nested list. This is useful for displaying the labels and values of smaller arrays or slice results::


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
