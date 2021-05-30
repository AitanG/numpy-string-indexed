from collections.abc import Iterable
import copy
import numpy as np

from . import constants

__all__ = ['ndarray']


class ndarray(object):
    '''
    A structure for matrix-like data, which stores the data as a classic Numpy
    ndarray, and provides the option to reference by human-readable values.

    The fm.ndarray class, plus the other functions exposed in the
    friendly_matrix package, are designed as substitutes for the Numpy ndarray,
    with comparable performance benchmarks and familiar, Numpy-style usage
    patterns.
    '''
    def __init__(self, array, dim_names=None, *args_dim_arrays,
                 **kwargs_dim_arrays):
        '''
        Performs argument validation and constructs an fm.ndarray instance.
        '''
        dim_arrays = ndarray.__validate_dim_arrays(
            array, dim_names, *args_dim_arrays, **kwargs_dim_arrays)

        if dim_arrays is None:
            # This means only arg was np.ndarray, so this is a casting
            self.__init_unlabeled(array)
        else:
            self._quick_init(array, dim_names, dim_arrays)

    def __init_unlabeled(array):
        '''
        Use normal 0-indexed integers for labels if none are provided.
        '''
        dim_names = [f'Axis {_}' for _ in range(array.ndim)]
        dim_arrays = [list(range(n)) for n in range(array.ndim)]
        self._quick_init(array, dim_names, dim_arrays)

    def _quick_init(self, array, dim_names, dim_arrays):
        '''
        Constructs an fm.ndarray instance.

        This method is separate from __init__() so that fm.ndarray objects can
        be constructed without first running validation (_new_ndarray()).

        Params:
            array:      np.ndarray containing the matrix values
            dim_names:  well-formed axis names
            dim_arrays: well-formed labels for each axis
        '''
        self.array = array
        self.dim_names = [str(n) for n in dim_names]
        self.dim_arrays = dim_arrays
        self.__init_index_map()
        self.__init_index_of_dim_map()
        self.__init_private_members()

    def dim_length(self, dim):
        '''
        Gets the length of the specified dim.

        Params:
            dim: label or integer index

        Returns: length
        '''
        return self.shape[self._to_dim_index(dim)]

    def __get_filter_criteria(self, *args, **kwargs):
        '''
        Gets a standardized dictionary of criteria to filter by.
        '''
        if kwargs:
            if not self.__is_kwarg_index_friendly:
                raise ValueError('ndarray instance is not kwargs-friendly')
            if args:
                raise ValueError('Args not accepted when using kwargs indices')
        else:
            is_dict = any(isinstance(arg, dict) for arg in args)
            if is_dict:
                if len(args) > 1:
                    raise ValueError('Extraneous arguments after filter dict')
                if len(args[0]) > self.ndim:
                    raise ValueError('Wrong number of dimensions provided'
                                     f' ({len(args[0])} > {self.ndim})')
            elif len(args) != self.ndim:
                raise ValueError('Wrong number of dimensions provided'
                                 f' ({len(args)} != {self.ndim})')

        # Format input into a dictionary of filter criteria
        criteria = {}
        if kwargs or is_dict:
            indices = kwargs or args[0]
            for dim_name in self.dim_names:
                if dim_name in indices:
                    criteria[dim_name] = indices[dim_name]
        else:
            for i, arg in enumerate(args):
                if arg is None:
                    continue
                criteria[self.dim_names[i]] = arg

        return criteria

    def __take_helper(self, *args, **kwargs):
        '''
        Facilitates code reuse between take() and take_A().
        '''
        criteria = self.__get_filter_criteria(*args, **kwargs)

        try:
            # Iteratively pare down the array
            result_array = self.array
            result_dim_names = copy.copy(self.dim_names)
            result_dim_arrays = []
            array_slice = []
            for i, dim_name in enumerate(self.dim_names):
                if dim_name in criteria:
                    if ndarray.__is_multiple_indices(criteria[dim_name]):
                        slice_indices = tuple(self.__index_map[dim_name][val]
                                              for val in criteria[dim_name])
                        result_dim_arrays.append([self.dim_arrays[i][j]
                                                  for j in slice_indices])
                    else:
                        # Remove the dimension if selecting by a single value
                        result_dim_names[i] = None
                        slice_indices = self.__index_map[dim_name][
                            criteria[dim_name]]
                else:
                    # No filtering on this dimension
                    slice_indices = slice(None)
                    result_dim_arrays.append(self.dim_arrays[i])

                array_slice = (slice(None),) * i + (slice_indices,)
                result_array = result_array[array_slice]

        except ValueError as ve:
            raise ValueError('Attempted to filter by unknown value')

        result_dim_names = [n for n in result_dim_names if n is not None]
        return result_dim_names, result_array, result_dim_arrays

    def take_A(self, *args, **kwargs):
        '''
        Same as take(), except returns only the array.
        '''
        _, result_array, _ = self.__take_helper(*args, **kwargs)
        return result_array

    def take(self, *args, **kwargs):
        '''
        Takes a slice of the array according to the specified indices

        Returns: a new ndarray object containing the filtered array
        '''
        dim_names, result_array, result_dim_arrays = self.__take_helper(
            *args, **kwargs)
        return _new_ndarray(result_array, dim_names, result_dim_arrays)

    def get(self, *args, **kwargs):
        '''
        Gets the singule value at the specified location

        Returns: the single value at the specified location
        '''
        array_slice = self.__get_array_slice(*args, **kwargs)

        # Index into the array to select the result
        result = self.array[array_slice]
        return result

    def set(self, val, *args, **kwargs):
        '''
        Sets the single value at the specified location

        Params:
            val: the value with which to replace the selected
        '''
        array_slice = self.__get_array_slice(*args, **kwargs)
        self.array[array_slice] = val

    def copy(self):
        '''
        Creates a deep copy of this object

        Returns: the copy
        '''
        array = copy.deepcopy(self.array)
        dim_names = copy.deepcopy(self.dim_names)
        dim_arrays = copy.deepcopy(self.dim_arrays)
        return _new_ndarray(array, dim_names, dim_arrays)

    def formatted(self, topological_order=None, formatter=None,
                  display_dim_names=True):
        '''
        Formats this fm.ndarray object in an embedded object notation

        Params:
            topological_order: which dims should be grouped together first
            formatter:         how each array value should be formatted
            display_dim_names: whether to display the dimension names

        Returns: the formatted results
        '''
        if topological_order is None:
            topological_order = self.dim_names
        elif not isinstance(topological_order, Iterable):
            raise ValueError('Topological order must be iterable')
        elif len(topological_order) < self.ndim:
            # Caller can specify just the first K dims in topological order
            unspecified_dim_names = [name for name in self.dim_names
                                     if name not in topological_order]
            topological_order += unspecified_dim_names

        # Construct a map from topological order index to dim array index
        dim_array_indices_in_top_order = []
        for i, dim_name in enumerate(topological_order):
            dim_array_indices_in_top_order.append(self.__index_of_dim[dim_name])

        if formatter is None:
            formatter = lambda v: str(v)

        # Recursively generate lines of output
        output_lines = []
        self.__formatted_helper(topological_order, formatter,
                               display_dim_names, output_lines,
                               dim_array_indices_in_top_order)

        return '\n'.join(output_lines)

    def __formatted_helper(self, topological_order, formatter,
                          display_dim_names, output_lines,
                          dim_array_indices_in_top_order,
                          *indices):
        '''
        Helper function for self.formatted() that populates output array

        Params:
            topological_order: which dims should be grouped together first
            formatter:         how each array value should be formatted
            display_dim_names: whether to display the dimension names
            output_lines:      the output array holding the result's text
            dim_array_indices_in_top_order:
                map from topological_order index to self.dim_array index
            indices:           human-readable indices for the current val
        '''
        n_indices = len(indices)
        if n_indices == len(topological_order):
            # We have all dim labels specified, output value
            indices_in_order = iter(indices[dim_array_indices_in_top_order[i]]
                                    for i in range(n_indices))
            val = self.get(*indices_in_order)
            formatted_val = str(formatter(val))
            output_lines.append(constants.INDENTATION * n_indices
                                + formatted_val)
        else:
            dim_array_index = dim_array_indices_in_top_order[n_indices]
            for dim_label in self.dim_arrays[dim_array_index]:
                # Display label
                label = ''
                if display_dim_names:
                    label += (topological_order[n_indices]
                              + constants.DIM_NAME_SEPARATOR)
                label += f'{dim_label}:'
                output_lines.append(constants.INDENTATION * n_indices + label)

                # Recurse
                next_indices = list(indices) + [dim_label]
                self.__formatted_helper(topological_order, formatter,
                                       display_dim_names, output_lines,
                                       dim_array_indices_in_top_order,
                                       *next_indices)

    def __get_array_slice(self, *args, **kwargs):
        '''
        Gets a tuple to be used to index into the underlying numpy array.
        This method essentially translates from human-readable indices
        to normal indices for use in get() and set().

        Params:
            args: the complete set of human-readable indices of the result

        Returns: a tuple representing a numpy array slice
        '''
        if kwargs:
            if not self.__is_kwarg_index_friendly:
                raise ValueError('ndarray instance is not kwargs-friendly')
            if args:
                raise ValueError('Args not accepted when using kwargs indices')
        else:
            is_dict = any(isinstance(arg, dict) for arg in args)
            if is_dict and len(args) > 1:
                raise ValueError('Extraneous arguments after filter dict')
            if ((is_dict and len(args[0]) != self.ndim)
                or len(args) != self.ndim):
                raise ValueError('Wrong number of dimensions provided'
                                 f' {len(args)} != {self.ndim})')

        try:
            # Construct a selector tuple to be applied to the np array
            if kwargs:
                indices = iter(kwargs[dim_name] for dim_name in self.dim_names)
            elif is_dict:
                indices = iter(args[0][dim_name] for dim_name in self.dim_names)
            else:
                indices = args
            array_slice_arr = []
            for dim_name, index in zip(self.dim_names, indices):
                select_index = self.__index_map[dim_name][index]
                array_slice_arr.append(select_index)

        except ValueError as ve:
            raise ValueError('Attempted to select by unknown value')

        return tuple(array_slice_arr)

    def __init_index_map(self):
        '''
        Initializes self.__index_map, which is used to translate between
        human-readable indices and array indices
        '''
        self.__index_map = {}
        for i, dim_name in enumerate(self.dim_names):
            dim_values = self.dim_arrays[i]
            value_to_index_map = {v: j for (j, v) in enumerate(dim_values)}
            self.__index_map[dim_name] = value_to_index_map

    def __init_index_of_dim_map(self):
        '''
        Initializes self.__index_of_dim, which precompuates the index of
        each dim name within self.dim_names.
        '''
        self.__index_of_dim = {n: i for (i, n) in enumerate(self.dim_names)}

    def __init_private_members(self):
        '''
        Sets helpful member variables:
            - Bool for whether all of this object's dim names are valid variable
              names. (If so, this object can be indexed using kwargs.)
        '''
        from keyword import iskeyword

        def is_valid_var_name(n):
            return n.isidentifier() and not iskeyword(n)

        self.__is_kwarg_index_friendly = all(is_valid_var_name(n)
                                             for n in self.dim_names)

    def _to_dim_index(self, dim):
        '''
        Determines the dim index corresponding to the provided dim

        Params:
            dim: a dim name, or a dim index

        Returns: the dim index
        '''
        dim_as_name = str(dim)
        if dim_as_name in self.dim_names:
            # If the arg is the name of a dimension, return its index
            return self.__index_of_dim[dim_as_name]

        if isinstance(dim, int):
            # If the arg is an int, treat it like an index
            return dim

        raise ValueError(f'Dim name \'{dim_as_name}\' not found in matrix')

    def _to_dim_indices(self, *dims):
        '''
        Determines the dim indices corresponding to the provided dims

        Params:
            dims: dim names or indices

        Returns: the dim indices
        '''
        return tuple(self._to_dim_index(d) for d in dims)

    @staticmethod
    def __is_multiple_indices(arg):
        '''
        Determines whether the provided value represents multiple indices

        Params:
            arg: the value to test

        Returns: whether the provided value represents multiple indices
        '''
        return isinstance(arg, Iterable) and not isinstance(arg, str)

    @staticmethod
    def __validate_dim_arrays(array, dim_names,
                              *args_dim_arrays,
                              **kwargs_dim_arrays):
        # First check for casting of np.ndarray
        if dim_names is None:
            if not isinstance(array, np.ndarray):
                raise TypeError(f'Unrecognized first argument')
            return None

        n_dim_names = len(dim_names)
        n_args_dim_arrays = len(args_dim_arrays)
        n_kwargs_dim_arrays = len(kwargs_dim_arrays)

        # Ensure dim_names are well-formed
        if array.ndim != n_dim_names:
            raise ValueError('Array has a different number of dimensions'
                             f' than dim_names ({array.ndim} !='
                             f' {n_dim_names})')
        if len(set(dim_names)) < n_dim_names:
            raise ValueError(f'Dim names must be unique')
        if any(isinstance(dim_name, int) for dim_name in dim_names):
            raise ValueError('Dim names must not be integers')

        # Parse args and kwargs
        named_dim_arrays = {}
        if n_args_dim_arrays > 0:
            # Dim arrays are being passed in as args, ensure no kwargs
            if n_kwargs_dim_arrays > 0:
                kwargs_keys = [repr(key) for key in kwargs_dim_arrays.keys()]
                display_kwargs_keys = ', '.join(kwargs_keys[:4])
                if n_kwargs_dim_arrays > 4:
                    display_kwargs_keys += (f', and {n_kwargs_dim_arrays - 4}'
                                            ' more')
                raise ValueError('Extraneous keyword arguments:'
                                 f' {display_kwargs_keys}')
            if isinstance(args_dim_arrays[0], dict):
                # Case 1: Dim arrays are being passed in via a dict
                if n_args_dim_arrays > 1:
                    raise ValueError('Extraneous arguments (expected 3, got'
                                     f' {2 + n_args_dim_arrays})')
                named_dim_arrays = args_dim_arrays[0]
            else:
                # Case 2: Dim arrays are being passed in sequentially
                if n_args_dim_arrays > n_dim_names:
                    raise ValueError('More dim arrays provided than dims'
                                     f' ({n_args_dim_arrays} > {n_dim_names})')
                named_dim_arrays = {dim_names[i]: args_dim_arrays[i]
                                    for i in range(n_args_dim_arrays)}

        elif n_kwargs_dim_arrays > 0:
            # Case 3: Dim arrays are being passed in as kwargs
            named_dim_arrays = kwargs_dim_arrays

        # Ensure named dim arrays are well-formed
        for dim_name, values in named_dim_arrays.items():
            if dim_name not in dim_names:
                raise ValueError(f'Unrecognized dim name \'{dim_name}\''
                                 ' provided as dim array')
            if not isinstance(values, Iterable):
                raise ValueError(f'Dim array for \'{dim_name}\' not iterable')

            values_list = values if isinstance(values, list) else list(values)
            named_dim_arrays[dim_name] = values_list

        # Convert named dim arrays object to ordered list
        ordered_dim_arrays = []
        for i, dim_name in enumerate(dim_names):
            if dim_name in named_dim_arrays:
                dim_array = named_dim_arrays[dim_name]
                if not len(dim_array) == len(set(dim_array)) == array.shape[i]:
                    raise ValueError(f'Lengths for dim {dim_name} don\'t match')
            else:
                # If no dim array is provided, matrix uses array indexing
                dim_array = list(range(array.shape[i]))
            ordered_dim_arrays.append(dim_array)

        # Ensure passed in strings are good for formatting
        for name_list in [dim_names] + ordered_dim_arrays:
            for name in name_list:
                if isinstance(name, str):
                    if '\n' in name:
                        raise ValueError(f'Newline characters not allowed in dim'
                                         ' names')
                    if name.strip() != name:
                        raise ValueError('Empty space not allowed for dim names:'
                                         f' {repr(name)}')

        return ordered_dim_arrays

    def __element_wise_operation(self, other, operator):
        '''
        Performs an element-wise operation between the arrays of two
        fm.ndarrays.

        Params:
            other:    the right operand
            operator: the operator to be used

        Returns: an fm.ndarray as the labeled result.
        '''
        if not isinstance(other, ndarray):
            raise TypeError('Element-wise operands must be ndarray type')
        if other.shape != self.shape:
            raise ValueError('Element-wise operands must have the same shape')
        if other.dim_names != self.dim_names:
            raise ValueError(f'Can\'t use \'{operator}\' operator on ndarray'
                             'instances with different dim names')
        if other.dim_arrays != self.dim_arrays:
            raise ValueError(f'Can\'t use \'{operator}\' operator on ndarray'
                             'instances with different dim arrays')

        result_array = getattr(self.array, operator)(other.array)
        dim_names = copy.copy(self.dim_names)
        dim_arrays = copy.copy(self.dim_arrays)
        return _new_ndarray(result_array, self.dim_names, self.dim_arrays)

    def __call__(self, *args, **kwargs):
        '''
        Caller can call an fm.ndarray object as shorthand for take() or get()
        '''
        if (len(args) == self.ndim
            and not any(ndarray.__is_multiple_indices(arg) for arg in args)):
            # All the provided filter values represent a single index
            return self.get(*args, **kwargs)

        # At least one of the provided filter values represents multiple
        # indices. This must be a take call
        return self.take(*args, **kwargs)

    def __str__(self):
        return str(self.array)

    def __lt__(self, other):
        return self.__element_wise_operation(other, '__lt__')

    def __le__(self, other):
        return self.__element_wise_operation(other, '__le__')

    def __gt__(self, other):
        return self.__element_wise_operation(other, '__gt__')

    def __ge__(self, other):
        return self.__element_wise_operation(other, '__ge__')

    def __add__(self, other):
        return self.__element_wise_operation(other, '__add__')

    def __sub__(self, other):
        return self.__element_wise_operation(other, '__sub__')

    def __mul__(self, other):
        return self.__element_wise_operation(other, '__mul__')

    def __matmul__(self, other):
        return self.__element_wise_operation(other, '__matmul__')

    def __truediv__(self, other):
        return self.__element_wise_operation(other, '__truediv__')

    def __floordiv__(self, other):
        return self.__element_wise_operation(other, '__floordiv__')

    def __mod__(self, other):
        return self.__element_wise_operation(other, '__mod__')

    def __divmod__(self, other):
        return self.__element_wise_operation(other, '__divmod__')

    def __pow__(self, other):
        return self.__element_wise_operation(other, '__pow__')

    def __lshift__(self, other):
        return self.__element_wise_operation(other, '__lshift__')

    def __rshift__(self, other):
        return self.__element_wise_operation(other, '__rshift__')

    def __and__(self, other):
        return self.__element_wise_operation(other, '__and__')

    def __xor__(self, other):
        return self.__element_wise_operation(other, '__xor__')

    def __or__(self, other):
        return self.__element_wise_operation(other, '__or__')

    def __radd__(self, other):
        return self.__element_wise_operation(other, '__radd__')

    def __rsub__(self, other):
        return self.__element_wise_operation(other, '__rsub__')

    def __rmul__(self, other):
        return self.__element_wise_operation(other, '__rmul__')

    def __rmatmul__(self, other):
        return self.__element_wise_operation(other, '__rmatmul__')

    def __rtruediv__(self, other):
        return self.__element_wise_operation(other, '__rtruediv__')

    def __rfloordiv__(self, other):
        return self.__element_wise_operation(other, '__rfloordiv__')

    def __rmod__(self, other):
        return self.__element_wise_operation(other, '__rmod__')

    def __rdivmod__(self, other):
        return self.__element_wise_operation(other, '__rdivmod__')

    def __rpow__(self, other):
        return self.__element_wise_operation(other, '__rpow__')

    def __rlshift__(self, other):
        return self.__element_wise_operation(other, '__rlshift__')

    def __rrshift__(self, other):
        return self.__element_wise_operation(other, '__rrshift__')

    def __rand__(self, other):
        return self.__element_wise_operation(other, '__rand__')

    def __rxor__(self, other):
        return self.__element_wise_operation(other, '__rxor__')

    def __ror__(self, other):
        return self.__element_wise_operation(other, '__ror__')

    def __iadd__(self, other):
        return self.__element_wise_operation(other, '__iadd__')

    def __isub__(self, other):
        return self.__element_wise_operation(other, '__isub__')

    def __imul__(self, other):
        return self.__element_wise_operation(other, '__imul__')

    def __imatmul__(self, other):
        return self.__element_wise_operation(other, '__imatmul__')

    def __itruediv__(self, other):
        return self.__element_wise_operation(other, '__itruediv__')

    def __ifloordiv__(self, other):
        return self.__element_wise_operation(other, '__ifloordiv__')

    def __imod__(self, other):
        return self.__element_wise_operation(other, '__imod__')

    def __ipow__(self, other):
        return self.__element_wise_operation(other, '__ipow__')

    def __ilshift__(self, other):
        return self.__element_wise_operation(other, '__ilshift__')

    def __irshift__(self, other):
        return self.__element_wise_operation(other, '__irshift__')

    def __iand__(self, other):
        return self.__element_wise_operation(other, '__iand__')

    def __ixor__(self, other):
        return self.__element_wise_operation(other, '__ixor__')

    def __ior__(self, other):
        return self.__element_wise_operation(other, '__ior__')

    def __neg__(self, other):
        return self.__element_wise_operation(other, '__neg__')

    def __pos__(self, other):
        return self.__element_wise_operation(other, '__pos__')

    def __abs__(self, other):
        return self.__element_wise_operation(other, '__abs__')

    def __invert__(self, other):
        return self.__element_wise_operation(other, '__invert__')

    def __eq__(self, other):
        return (isinstance(other, ndarray)
                and np.all(self.array == other.array)
                and self.dim_names == other.dim_names
                and self.dim_arrays == other.dim_arrays)

    def __getitem__(self, *args):
        return self.array.__getitem__(*args)

    def __setitem__(self, *args):
        return self.array.__setitem__(*args)


    '''
    ===========================================================================
                                Numpy-like methods
    ===========================================================================
    '''

    '''
    Array properties
    '''

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size
    
    @property
    def dtype(self):
        return self.array.dtype
    
    @property
    def itemsize(self):
        return self.array.itemsize
    

    '''
    Transpose-like operations
    '''

    def moveaxis_A(self, dim, new_dim):
        '''
        Same as moveaxis(), except returns only the array.
        '''
        dim_index, new_dim_index = self._to_dim_indices(dim, new_dim)
        return np.moveaxis(self.array, dim_index, new_dim_index)

    def moveaxis(self, dim, new_dim):
        '''
        Performs a Numpy-style moveaxis operation on the fm.ndarray.

        Params:
            dim:     the dim to move
            new_dim: the dim whose place `dim` will take

        Returns: fm.ndarray
        '''
        dim_index, new_dim_index = self._to_dim_indices(dim, new_dim)
        array = np.moveaxis(self.array, dim_index, new_dim_index)

        dim_names = copy.copy(self.dim_names)
        dim_arrays = list(copy.copy(self.dim_arrays))
        dim_names.insert(new_dim_index, dim_names.pop(dim_index))
        dim_arrays.insert(new_dim_index, dim_arrays.pop(dim_index))

        return _new_ndarray(array, dim_names, dim_arrays)

    def swapaxes_A(self, dim1, dim2):
        '''
        Same as swapaxes(), except returns only the array.
        '''
        dim_index1, dim_index2 = self._to_dim_indices(dim1, dim2)
        return self.array.swapaxes(dim_index1, dim_index2)

    def swapaxes(self, dim1, dim2):
        '''
        Performs a Numpy-style swapaxes operation on the fm.ndarray.

        Params:
            dim1: first dim
            dim2: other dim

        Returns: fm.ndarray
        '''
        dim_index1, dim_index2 = self._to_dim_indices(dim1, dim2)
        array = self.array.swapaxes(dim_index1, dim_index2)

        dim_names = copy.copy(self.dim_names)
        dim_arrays = list(copy.copy(self.dim_arrays))
        temp = dim_names[dim_index1]
        dim_names[dim_index1] = dim_names[dim_index2]
        dim_names[dim_index2] = temp
        temp = dim_arrays[dim_index1]
        dim_arrays[dim_index1] = dim_arrays[dim_index2]
        dim_arrays[dim_index2] = temp

        return _new_ndarray(array, dim_names, dim_arrays)

    def transpose_A(self):
        '''
        Same as swapaxes(), except returns only the array.
        '''
        return self.moveaxis_A(1, 0)

    def transpose(self):
        '''
        Performs a Numpy-style transpose operation on the fm.ndarray.

        Returns: fm.ndarray
        '''
        return self.moveaxis(1, 0)

    @property
    def T_A(self):
        return self.transpose_A()

    @property
    def T(self):
        return self.transpose()


    '''
    Aggregating across a dimension
    '''

    def ___aggregation_result_A(self, axis, aggregate_fn):
        '''
        Same as __aggregation_result(), except returns only the array.
        '''
        if axis is None:
            return aggregate_fn(self.array, axis=axis)
        dim_index = self._to_dim_index(axis)
        return aggregate_fn(self.array, axis=dim_index)

    def __aggregation_result(self, axis, aggregate_fn):
        '''
        Applies an aggregation function to the fm.ndarray.

        Params:
            axis:         the dim over which to aggregate
            aggregate_fn: the Numpy aggregation function

        Returns: fm.ndarray
        '''
        if axis is None:
            return aggregate_fn(self.array, axis=axis)
        dim_index = self._to_dim_index(axis)
        array = aggregate_fn(self.array, axis=dim_index)

        dim_names = copy.copy(self.dim_names)
        dim_arrays = list(copy.copy(self.dim_arrays))
        dim_names.pop(dim_index)
        dim_arrays.pop(dim_index)

        return _new_ndarray(array, dim_names, dim_arrays)

    def mean_A(self, axis=None):
        '''
        Same as mean(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.mean)

    def mean(self, axis=None):
        '''
        Compuates the average along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.mean)

    def std_A(self, axis=None):
        '''
        Same as std_A(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.std)

    def std(self, axis=None):
        '''
        Compuates the standard deviation along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.std)

    def var_A(self, axis=None):
        '''
        Same as var(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.var)

    def var(self, axis=None):
        '''
        Compuates the variance along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.var)

    def sum_A(self, axis=None):
        '''
        Same as sum(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.sum)

    def sum(self, axis=None):
        '''
        Computes the average along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.sum)

    def prod_A(self, axis=None):
        '''
        Same as prod(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.prod)

    def prod(self, axis=None):
        '''
        Computes the product along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.prod)

    def min_A(self, axis=None):
        '''
        Same as min(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.min)

    def min(self, axis=None):
        '''
        Computes the minimum along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.min)

    def argmin_A(self, axis=None):
        '''
        Same as argmin(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.argmin)

    def argmin(self, axis=None):
        '''
        Computes the index of the minimum along an axis.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.argmin)

    def all_A(self, axis=None):
        '''
        Same as all(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.all)

    def all(self, axis=None):
        '''
        Computes whether all of the values along an axis are truthy.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.all)

    def any_A(self, axis=None):
        '''
        Same as any(), except returns only the array.
        '''
        return self.___aggregation_result_A(axis, np.any)

    def any(self, axis=None):
        '''
        Computes whether any of the values along an axis are truthy.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        return self.__aggregation_result(axis, np.any)

    def cumsum_A(self, axis=0):
        '''
        Same as cumsum(), except returns only the array.
        '''
        dim_index = self._to_dim_index(axis)
        return self.array.cumsum(axis=dim_index)

    def cumsum(self, axis=0):
        '''
        Computes the cumulative sum along an axis. Does not change the
        fm.ndarray's dimensionality.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        dim_index = self._to_dim_index(axis)
        array = self.array.cumsum(axis=dim_index)
        return _new_ndarray(array, self.dim_names, self.dim_arrays)

    def cumprod_A(self, axis=0):
        '''
        Same as cumprod(), except returns only the array.
        '''
        dim_index = self._to_dim_index(axis)
        return self.array.cumprod(axis=dim_index)

    def cumprod(self, axis=0):
        '''
        Computes the cumulative product along an axis. Does not change the
        fm.ndarray's dimensionality.

        Params:
            axis: dim or dim index over which to aggregate

        Returns: fm.ndarray
        '''
        dim_index = self._to_dim_index(axis)
        array = self.array.cumprod(axis=dim_index)
        return _new_ndarray(array, self.dim_names, self.dim_arrays)

    def __squeeze_helper(self):
        '''
        Facilitates code reuse between squeeze() and squeeze_A().
        '''
        dim_names = copy.copy(self.dim_names)
        dim_arrays = copy.copy(self.dim_arrays)

        array_slice = [slice(None)] * self.ndim
        for i, dim_len in enumerate(self.shape):
            if dim_len == 1:
                array_slice[i] = 0
                dim_names.pop(i)
                dim_arrays.pop(i)
        array = self.array[tuple(array_slice)]

        return array, dim_names, dim_arrays

    def squeeze_A(self):
        '''
        Same as squeeze(), except returns only the array.
        '''
        array, _, _ = self.__squeeze_helper()
        return array

    def squeeze(self):
        '''
        Aggregates over (removes) any length 1 dimensions.
        '''
        args = self.__squeeze_helper()
        return _new_ndarray(*args)

    @property
    def A(self):
        '''
        Shorthand for extracting just the np.ndarray from an fm.ndarray.

        Example:
            cov = compute_covariance(old.A, new.A)
        '''
        return self.array


class Empty:
    '''
    Empty class used to allow the creation of an fm.ndarray instance without
    calling __init__().
    '''
    pass


def _new_ndarray(*args):
    '''
    Substitute initializer for fm.ndarray that avoids performing validation on
    fm.ndarray objects created internally, by avoiding calling __init__().
    '''
    result = Empty()
    result.__class__ = ndarray
    result._quick_init(*args)
    return result
