.. title:: NumPy String-Indexed


NumPy String-Indexed
====================
NumPy String-Indexed is a `NumPy <https://numpy.org/>`_ extension that allows arrays to be indexed using descriptive string labels, rather than conventional zero-indexing. When an ``ndarray`` (AKA a friendly matrix) instance is initialized, labels are assigned to each array index and each dimension, and they stick to the array after NumPy-style operations such as transposing, concatenating, and aggregating. This prevents Python programmers from having to keep track mentally of what each axis and each index represents, instead making each reference to the array in code naturally self-documenting.

NumPy String-Indexed is especially useful for applications like machine learning, scientific computing, and data science, where there is heavy use of multidimensional arrays.

The friendly matrix object is implemented as a lightweight wrapper around a NumPy ``ndarray``. It's easy to add to a new or existing project to make it easier to maintain code, and has negligible memory and performance overhead compared to the size of array (*O(x + y + z)* vs. *O(xyz)*).


Basic functionality
-------------------

It's recommended to import NumPy String-Indexed idiomatically as ``fm``::

	import friendly_matrix as fm

Labels are provided during object construction and can optionally be used in place of numerical indices for slicing and indexing.

The example below shows how to construct a friendly matrix containing an image with three color channels::

	image = fm.ndarray(
		numpy_ndarray_image,  # np.ndarray with shape (3, 100, 100)
		dim_names=['color_channel', 'top_to_bottom', 'left_to_right'],
		color_channel=['R', 'G', 'B'])

The array can then be sliced like this::

	# friendly matrix with shape (100, 100)
	r_channel = image(color_channel='R')

	# an integer
	g_top_left_pixel_value = image('G', 0, 0)

	# friendly matrix with shape (100, 50)
	br_channel_left_half = image(
		color_channel=('B', 'R'),
		left_to_right=range(image.dim_length('left_to_right') // 2))


Installation
------------

::

	pip install numpy-string-indexed

NumPy String-Indexed is listed in `PyPI <https://pypi.org/project/numpy-string-indexed/>`_ and can be installed with ``pip``.

**Prerequisites**: NumPy String-Indexed 0.0.1 requires Python 3 and a compatible installation of the `NumPy <https://pypi.org/project/numpy/>`_ Python package.


Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guide
   basic
   operations
   compute
   format

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Discussion and support
----------------------

NumPy String-Indexed is available under the `MIT License <https://opensource.org/licenses/MIT>`_.
