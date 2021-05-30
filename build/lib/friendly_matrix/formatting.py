import numpy as np

from . import constants
from . import friendly_matrix as fm

__all__ = ['formatted', 'from_formatted_A', 'from_formatted']


def formatted(friendly, topological_order=None, formatter=None,
              display_dim_names=True):
    '''
    Formats the provided fm.ndarray object in an embedded object notation.

    Params:
        friendly:          the fm.ndarray to format
        topological_order: which dims should be grouped together first
        formatter:         how each array value should be formatted
        display_dim_names: whether to display the dimension names

    Returns: the formatted results
    '''
    return friendly.formatted(topological_order, formatter, display_dim_names)


def from_formatted_A(formatted_friendly, dtype=np.str):
    '''
    Same as from_formatted(), except returns only the array.
    '''
    friendly = from_formatted(formatted_friendly, dtype)
    return friendly.array


def from_formatted(formatted_friendly, dtype=np.str):
    '''
    Constructs an fm.ndarray object out of its string representation. Assumes a
    valid string is provided.

    Params:
        formatted_friendly: the string representation

    Returns: fm.ndarray object
    '''
    lines = formatted_friendly.split('\n')

    # First, figure out ndims
    ndims = 0
    prev_n_indentation_chars = -constants.INDENTATION_LEN
    for line in lines:
        n_indentation_chars = __get_n_indentation_chars(line)

        if n_indentation_chars > prev_n_indentation_chars:
            ndims += 1
            prev_n_indentation_chars = n_indentation_chars
        else:
            # We're at the line after the first actual value
            ndims -= 1
            break

    # Second, discover whether dim names are provided
    n_indentation_chars_for_each_line = []
    dim_names = [None] * ndims
    dim_names_are_provided = True
    for line in lines:
        n_indentation_chars = __get_n_indentation_chars(line)

        # Store n_indentation_chars for later
        n_indentation_chars_for_each_line.append(n_indentation_chars)

        dim_index = n_indentation_chars // constants.INDENTATION_LEN
        if dim_index < ndims:
            # This is a label, not a value
            if constants.DIM_NAME_SEPARATOR not in line:
                # If a single label has no =, we know dim names are not given
                dim_names_are_provided = False
                break
            # If a single label has a =, this may or may not be the separator
            # for the dim name
            supposed_dim_name = line[n_indentation_chars:].split(
                constants.DIM_NAME_SEPARATOR)[0]
            if dim_names[dim_index] is not None:
                if dim_names[dim_index] != supposed_dim_name:
                    # Supposed dim names don't match up, so we know dim names
                    # are not given
                    dim_names_are_provided = False
                    break
            else:
                dim_names[dim_index] = supposed_dim_name

    if not dim_names_are_provided:
        dim_names = list(range(ndims))

    # Fourth, extract values and dim_arrays
    values_1d = []
    dim_arrays = [[] for _ in range(ndims)]
    for i, line in enumerate(lines):
        # Get n_indentation_chars and dim_index
        if i < len(n_indentation_chars_for_each_line):
            n_indentation_chars = n_indentation_chars_for_each_line[i]
        else:
            n_indentation_chars = __get_n_indentation_chars(line)
        if n_indentation_chars > ndims * constants.INDENTATION_LEN:
            n_indentation_chars = ndims * constants.INDENTATION_LEN
        dim_index = n_indentation_chars // constants.INDENTATION_LEN

        if dim_index < ndims:
            # This is a dim label
            if dim_names_are_provided:
                dim_label = ''.join(line[n_indentation_chars:-1].split(
                    constants.DIM_NAME_SEPARATOR)[1:])
            else:
                dim_label = line[n_indentation_chars:-1]

            if dim_label not in dim_arrays[dim_index]:
                # Naively append the dim label if it's a new one
                dim_arrays[dim_index].append(dim_label)
        else:
            # This is a value
            value = dtype(line[n_indentation_chars:])
            values_1d.append(value)

    shape = [len(dim_array) for dim_array in dim_arrays]
    array = np.array(values_1d).reshape(shape)

    return fm._new_ndarray(array, dim_names, dim_arrays)


def __get_n_indentation_chars(line):
    '''
    Helper function that counts the number of indentation chars in a line.

    Params:
        line: the input string

    Returns: the number of indentation chars
    '''
    n_indentation_chars = 0
    while True:
        chars = line[n_indentation_chars:(n_indentation_chars
                                          + constants.INDENTATION_LEN)]
        if chars != constants.INDENTATION:
            break
        n_indentation_chars += constants.INDENTATION_LEN

    return n_indentation_chars