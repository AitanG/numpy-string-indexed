import copy
import numpy as np

from . import friendly_matrix as fm

__all__ = ['moveaxis_A', 'moveaxis', 'swapaxes_A', 'swapaxes', 'transpose_A', 'transpose', 'concatenate_A', 'concatenate', 'stack_A', 'stack', 'vstack_A', 'vstack', 'hstack_A', 'hstack', 'flip_A', 'flip', 'fliplr_A', 'fliplr', 'flipud_A', 'flipud', 'mean_A', 'mean', 'std_A', 'std', 'var_A', 'var', 'sum_A', 'sum', 'prod_A', 'prod', 'min_A', 'min', 'argmin_A', 'argmin', 'all_A', 'all', 'any_A', 'any', 'cumsum_A', 'cumsum', 'cumprod_A', 'cumprod', 'squeeze_A', 'squeeze']


'''
Transpose-like operations
'''

def moveaxis_A(friendly, dim, new_dim):
    return friendly.moveaxis_A(dim, new_dim)


def moveaxis(friendly, dim, new_dim):
    return friendly.moveaxis(dim, new_dim)


def swapaxes_A(friendly, dim1, dim2):
    return friendly.swapaxes_A(dim1, dim2)


def swapaxes(friendly, dim1, dim2):
    return friendly.swapaxes(dim1, dim2)


def transpose_A(friendly):
    return friendly.transpose_A()


def transpose(friendly):
    return friendly.transpose()


'''
Joining arrays
'''

def concatenate_A(friendlies, axis=0):
    '''
    Same as concatenate(), except returns only the array.
    '''
    dim_index = friendlies[0]._to_dim_index(axis)
    arrays = tuple(friendly.array for friendly in friendlies)
    return np.concatenate(arrays, dim_index)


def concatenate(friendlies, axis=0):
    '''
    Concatenates the provided fm.ndarrays along the provided axis.

    Params:
        friendlies: the fm.ndarray to concatenate
        axis:       the axis to concatenate along

    Returns: fm.ndarray
    '''
    dim_index = friendlies[0]._to_dim_index(axis)
    arrays = tuple(friendly.array for friendly in friendlies)
    array = np.concatenate(arrays, dim_index)

    # Ensure that we're joining along the same dim name for each friendly
    dim_name_first = friendlies[0].dim_names[dim_index]
    for friendly in friendlies[1:]:
        dim_name_cur = friendly.dim_names[dim_index]
        if dim_name_cur != dim_name_first:
            raise ValueError(f'Different dim names for axis {dim_index}'
                             f' across different ndarrays: {dim_name_first}'
                             f' and {dim_name_cur}.')

    # Update array of values for the axis being extended
    extended_dim_array = []
    for friendly in friendlies:
        extended_dim_array += friendly.dim_arrays[dim_index]

    dim_arrays = list(copy.copy(friendlies[0].dim_arrays))
    dim_arrays[dim_index] = extended_dim_array

    return fm._new_ndarray(array, friendlies[0].dim_names, dim_arrays)


def stack_A(friendlies, axis_name=None, axis_array=None, axis=None):
    '''
    Same as stack(), except returns only the array.
    '''
    if axis is None:
        if axis_name is not None:
            raise ValueError(f'Specified new axis name but no axis')
        axis = 0
    dim_index = friendlies[0]._to_dim_index(axis)
    arrays = tuple(friendly.array for friendly in friendlies)
    return np.stack(arrays, dim_index)


def stack(friendlies, axis_name, axis_array, axis=0):
    '''
    Stacks the provided fm.ndarrays along a new axis.

    Params:
        friendlies: the fm.ndarrays to concatenate
        axis_name:  the name of the newly created axis
        axis_array: the labels for the newly created axis
        axis:       the location of the new axis

    Returns: fm.ndarray
    '''
    if len(axis_array) != len(friendlies):
        raise ValueError(f'Axis array must be the same length as the number'
                         f' of ndarrays being stacked ({len(axis_array)} !='
                         f' {len(friendlies)})')

    dim_index = friendlies[0]._to_dim_index(axis)
    arrays = tuple(friendly.array for friendly in friendlies)
    array = np.stack(arrays, dim_index)

    dim_names_first = friendlies[0].dim_names
    for friendly in friendlies[1:]:
        dim_names_cur = friendly.dim_names
        if dim_names_cur != dim_names_first:
            raise ValueError('All ndarrays being stacked must have the'
                             ' same dim names')

    dim_names = copy.copy(friendlies[0].dim_names)
    dim_arrays = list(copy.copy(friendlies[0].dim_arrays))
    dim_names.insert(dim_index, axis_name)
    dim_arrays.insert(dim_index, axis_array)

    return fm._new_ndarray(array, dim_names, dim_arrays)


def vstack_A(friendlies):
    '''
    Same as vstack(), except returns only the array.
    '''
    if len(friendlies[0].shape) == 1:
        # Vstacking 1-dimensional arrays requires creating a new dimension
        raise ValueError('Can\'t perform vstack() on one-dimensional'
                         ' ndarrays. Use stack() instead')

    return concatenate_A(friendlies, axis=0)


def vstack(friendlies):
    '''
    Vertically stacks the provided fm.ndarrays.

    Params:
        friendlies: the fm.ndarrays to stack

    Returns: fm.ndarray
    '''
    if len(friendlies[0].shape) == 1:
        # Vstacking 1-dimensional arrays requires creating a new dimension
        raise ValueError('Can\'t perform vstack() on one-dimensional'
                         ' ndarrays. Use stack() instead')

    return concatenate(friendlies, axis=0)


def hstack_A(friendlies):
    '''
    Same as hstack(), except returns only the array.
    '''
    if len(friendlies[0].shape) == 1:
        return concatenate_A(friendlies, axis=0)

    return concatenate_A(friendlies, axis=1)


def hstack(friendlies):
    '''
    Horizontally stacks the provided fm.ndarrays.

    Params:
        friendlies: the fm.ndarrays to stack

    Returns: fm.ndarray
    '''
    if len(friendlies[0].shape) == 1:
        return concatenate(friendlies, axis=0)

    return concatenate(friendlies, axis=1)


'''
Rearranging elements
'''

def flip_A(friendly, axis=None):
    '''
    Same as flip(), except returns only the array.
    '''
    if axis is None:
        axes_to_flip = list(range(friendly.ndim))
    else:
        dim_index = friendly._to_dim_index(axis)
        axes_to_flip = [dim_index]

    return np.flip(friendly.array, axis)


def flip(friendly, axis=None):
    '''
    Reverses the order of elements along the provided axes.

    Params:
        friendly: the fm.ndarray to flip
        axis:     the axis along which to flip

    Returns: fm.ndarray
    '''
    if axis is None:
        axes_to_flip = list(range(friendly.ndim))
    else:
        dim_index = friendly._to_dim_index(axis)
        axes_to_flip = [dim_index]

    array = np.flip(friendly.array, axis)

    dim_arrays = list(copy.copy(friendly.dim_arrays))
    for dim_index in axes_to_flip:
        dim_arrays[dim_index] = list(reversed(dim_arrays[dim_index]))

    return fm._new_ndarray(array, friendly.dim_names, dim_arrays)


def fliplr_A(friendly):
    '''
    Same as fliplr(), except returns only the array.
    '''
    return flip_A(friendly, axis=0)


def fliplr(friendly):
    '''
    Reverses the order of elements along the first axis.

    Params:
        friendly: the fm.ndarray to flip

    Returns: fm.ndarray
    '''
    return flip(friendly, axis=0)


def flipud_A(friendly):
    '''
    Same as flipud(), except returns only the array.
    '''
    return flip_A(friendly, axis=1)


def flipud(friendly):
    '''
    Reverses the order of elements along the second axis.

    Params:
        friendly: the fm.ndarray to flip

    Returns: fm.ndarray
    '''
    return flip(friendly, axis=1)


'''
Aggregating across a dimension
'''

def mean_A(friendly, axis=0):
    return friendly.mean_A(axis)


def mean(friendly, axis=0):
    return friendly.mean(axis)


def std_A(friendly, axis=0):
    return friendly.std_A(axis)


def std(friendly, axis=0):
    return friendly.std(axis)


def var_A(friendly, axis=0):
    return friendly.var_A(axis)


def var(friendly, axis=0):
    return friendly.var(axis)


def sum_A(friendly, axis=0):
    return friendly.sum_A(axis)


def sum(friendly, axis=0):
    return friendly.sum(axis)


def prod_A(friendly, axis=0):
    return friendly.prod_A(axis)


def prod(friendly, axis=0):
    return friendly.prod(axis)


def min_A(friendly, axis=0):
    return friendly.min_A(axis)


def min(friendly, axis=0):
    return friendly.min(axis)


def argmin_A(friendly, axis=0):
    return friendly.argmin_A(axis)


def argmin(friendly, axis=0):
    return friendly.argmin(axis)


def all_A(friendly, axis=0):
    return friendly.all_A(axis)


def all(friendly, axis=0):
    return friendly.all(axis)


def any_A(friendly, axis=0):
    return friendly.any_A(axis)


def any(friendly, axis=0):
    return friendly.any(axis)


def cumsum_A(friendly, axis=0):
    return friendly.cumsum_A(axis)


def cumsum(friendly, axis=0):
    return friendly.cumsum(axis)


def cumprod_A(friendly, axis=0):
    return friendly.cumprod_A(axis)


def cumprod(friendly, axis=0):
    return friendly.cumprod(axis)


def squeeze_A(friendly):
    return friendly.squeeze_A()


def squeeze(friendly):
    return friendly.squeeze()
