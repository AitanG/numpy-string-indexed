.. title:: Formatting arrays for output


Formatting arrays for output
============================

``friendly_matrix.ndarray`` methods
-----------------------------------

.. py:class:: friendly_matrix.ndarray
	:noindex:

	.. py:method:: friendly_matrix.ndarray.formatted([topological_order=None[, formatter=None[, display_dim_names=True]]]) -> str

		Formats the ``friendly_matrix.ndarray`` instance as a nested list. All elements in the array are listed linearly under their dim index labels. The order in which dimensions are traversed can be set, as well as whether dim names are displayed alongside dim index labels, and how elements should be formatted before being appended to the result.

		This is useful for displaying the labels and values of smaller matrices or slice results.

		:param topological_order: iterable representing the order in which dimensions should be traversed for output
		:param formatter: callable that formats an element for output
		:param display_dim_names: whether to display dim names alongside dim array labels

		Example usage::

			prices.formatted(topological_order=["Year", "Size"],
							 formatter=price_formatter,
							 display_dim_names=True)

			'''
			Example output:

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
			'''

Module functions
----------------

.. py:function:: friendly_matrix.formatted(friendly[, topological_order=None[, formatter=None[, display_dim_names=True]]]) -> str

	Equivalent to `friendly.formatted(topological_order, formatter, display_dim_names)`.

.. py:function:: friendly_matrix.from_formatted(formatted_friendly[, dtype=numpy.str]) -> friendly_matrix.ndarray

	Deserializes a string representation of a ``friendly_matrix.ndarray`` instance back into a new ``friendly_matrix.ndarray`` instance.

	:param formatted_friendly: the formatted ``friendly_matrix.ndarray`` instance
	:param dtype: the data type of the result `friendly_matrix.ndarray`

	Assumes a valid string is provided.

	:return: The new ``friendly_matrix.ndarray`` instance

.. py:function:: friendly_matrix.from_formatted_A(formatted_friendly[, dtype=numpy.str]) -> friendly_matrix.ndarray

	Same as :py:func:`friendly_matrix.from_formatted()`, except returns only the NumPy array.