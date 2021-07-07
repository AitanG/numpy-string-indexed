from collections.abc import Iterable
import itertools
import numpy as np

from . import friendly_matrix as fm

__all__ = ['compute_ndarray', 'compute_ndarray_A']


def compute_ndarray(dim_names, *args, dtype=np.float32):
    '''
    Generates a `friendly_matrix.ndarray` object by computing it and having it
    indexable the same way it's computed: using embedded loops over
    human-readable lists of values.

    Params:
        `dim_names`: The name of each dimension, which should match args order
        `args`:      Iterables or callables specifying how to calculate results
        `dtype`:     The data type of the computed results

    Returns: The resulting `friendly_matrix.ndarray` object
    '''
    dim_arrays = [arg for arg in args if not callable(arg)]
    shape = tuple(len(arg) for arg in dim_arrays)
    array = np.empty(shape, dtype=dtype)
    friendly = fm.ndarray(array, dim_names, *dim_arrays)
    __compute_ndarray_helper(friendly, (), *args)

    return friendly


def compute_ndarray_A(dim_names, *args, dtype=np.float32):
    '''
    Same as `compute_ndarray()`, except returns only the array.
    '''
    dim_arrays = [arg for arg in args if not callable(arg)]
    shape = tuple(len(arg) for arg in dim_arrays)
    array = np.empty(shape, dtype=dtype)
    friendly = fm.ndarray(array, list(range(len(dim_arrays))), *dim_arrays)
    __compute_ndarray_helper(friendly, (), *args)

    return friendly.array


def __compute_ndarray_helper(friendly, indices_already_set,
                             *args, **intermediate_results):
    '''
    Recursive helper function for `compute_ndarray()` which executes any
    intermediate callables while looping over all combinations of index
    values, before finally computing the result for each element.

    Params:
        `friendly`:             The fm.ndarray object to update
        `indices_already_set`:  Tuple containing the current index values
        `args`:                 The (remaining) args from `compute_ndarray()`
        `intermediate_results`: Dict containing results from upstream functions
    '''
    # Iterate through args until we find a callable
    cur_arrays = []
    fn = None
    for arg in args:
        is_callable = callable(arg)
        is_iterable = isinstance(arg, Iterable)
        if is_callable and is_iterable:
            raise ValueError('Args should not be both callable and iterable')
        elif is_callable:
            fn = arg
            break
        elif is_iterable:
            cur_arrays.append(arg)
        else:
            raise ValueError('Args should be either callable or iterable')

    if fn is None:
        # We reached the end of the args without seeing a callable
        raise ValueError('Last arg is not callable')

    n_indices_newly_set = len(cur_arrays)
    if n_indices_newly_set == 0:
        # We're seeing a callable without an iterable before it.
        # The below line makes this case work with looping.
        X = [()]
    else:
        # We want to loop through each combination of the current values
        X = itertools.product(*cur_arrays)

    if n_indices_newly_set == len(args) - 1:
        # Base case: we've reached the last argument.
        # The result of this callable is what we stick in the array.
        for x in X:
            indices = indices_already_set + x
            result = fn(*indices, **intermediate_results)

            # Here's where the magic happens: friendly is indexed by
            # human-readable values--here, indices
            friendly.set(result, *indices)
    else:
        for x in X:
            indices = indices_already_set + x
            remaining_args = args[(n_indices_newly_set + 1):]
            new_intermediate_results = fn(*indices, **intermediate_results)
            if not isinstance(new_intermediate_results, dict):
                new_intermediate_results = {}
            updated_intermediate_results = {
                **intermediate_results,
                **new_intermediate_results
            }

            __compute_ndarray_helper(friendly, indices, *remaining_args,
                                     **updated_intermediate_results)
