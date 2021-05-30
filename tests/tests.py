import copy
import numpy as np
import os
import pdb
import unittest

import friendly_matrix as fm
from friendly_matrix.compute_matrix import *


class TestObjectConstruction(unittest.TestCase):
    def setUp(self):
        self.makes = ['A, Inc.', 'B, Inc.']
        self.colors = ['red', 'blue', 'green']
        self.sizes = ['small', 'big']
        self.years = [2000, 2020]
        self.price_array = np.arange(24).reshape(2, 3, 2, 2)

    def test_valid(self):
        friendly = fm.ndarray(self.price_array,
                              ['Make', 'Color', 'Size', 'Year'],
                              self.makes, self.colors, self.sizes, self.years)

        self.assertEqual(friendly.shape, (2, 3, 2, 2))
        self.assertEqual(friendly.ndim, 4)
        self.assertEqual(friendly.dim_names, ['Make', 'Color', 'Size', 'Year'])
        self.assertEqual(friendly.dim_arrays[0], self.makes)
        self.assertEqual(friendly.dim_arrays[3], self.years)
        self.assertTrue(np.all(friendly.array == self.price_array))

    def test_invalid_dim_array(self):
        try:
            makes_duplicate = copy.copy(self.makes)
            makes_duplicate[1] = makes_bad[0]
            friendly = fm.ndarray(self.price_array,
                                  ['Make', 'Color', 'Size', 'Year'],
                                  makes_duplicate, self.colors, self.sizes, self.years)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        try:
            makes_wrong_size = copy.copy(self.makes)
            makes_wrong_size.append('C, Inc.')
            friendly = fm.ndarray(self.price_array,
                                  ['Make', 'Color', 'Size', 'Year'],
                                  makes_wrong_size, self.colors, self.sizes, self.years)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        try:
            # Extra dim array
            friendly = fm.ndarray(self.price_array,
                                  ['Make', 'Color', 'Size', 'Year'],
                                  self.makes, self.colors, self.sizes, self.years, [])
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_invalid_array(self):
        try:
            price_array_wrong_size = self.price_array[1:]
            friendly = fm.ndarray(price_array_wrong_size,
                                  ['Make', 'Color', 'Size', 'Year'],
                                  self.makes, self.colors, self.sizes, self.years)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        try:
            price_array_wrong_ndim = self.price_array[0]
            friendly = fm.ndarray(price_array_wrong_ndim,
                                  ['Make', 'Color', 'Size', 'Year'],
                                  self.makes, self.colors, self.sizes, self.years)
            self.assertTrue(False)
        except:
            self.assertTrue(True)


class TestComputation(unittest.TestCase):
    def setUp(self):
        self.makes = ['A, Inc.', 'B, Inc.']
        self.colors = ['red', 'blue', 'green']
        self.sizes = ['small', 'big']
        self.years = [2000, 2020]
        self.golden_array = np.array(
            [[[[0.0, 17.0], [0.0, 19.0]],
              [[0.0, 17.0], [0.0, 19.0]],
              [[0.0, 17.0], [0.0, 19.0]]],
             [[[9.0, 17.0], [11.0, 19.0]],
              [[0.0, 0.0], [0.0, 0.0]],
              [[10.800000190734863, 18.0], [12.600000381469727, 19.799999237060547]]]])

        self.golden_friendly = fm.ndarray(self.golden_array,
                                          ['Make', 'Color', 'Size', 'Year'],
                                          self.makes, self.colors, self.sizes, self.years)

    def test_simple(self):
        def calculate_price(make, color, size, year):
            i = self.makes.index(make)
            j = self.colors.index(color)
            k = self.sizes.index(size)
            l = self.years.index(year)

            return self.golden_array[i, j, k, l]

        friendly = compute_ndarray(['Make', 'Color', 'Size', 'Year'],
                                   self.makes,
                                   self.colors,
                                   self.sizes,
                                   self.years,
                                   calculate_price,
                                   dtype=np.float32)

        self.assertEqual(friendly, self.golden_friendly)

    def test_complex(self):
        def calculate_bulk_discount(make, color):
            result = {
                'bulk_discount': 0,
                'is_in_stock': True,
            }
            if make == 'A, Inc.' or color == 'red':
                result['bulk_discount'] = 3
            if make == 'B, Inc.' and color == 'blue':
                result['is_in_stock'] = False
            return result

        def calculate_promo_discount(make, color, **kwargs):
            result = {
                'promo_discount': 0.1
            }
            if kwargs['bulk_discount']:
                result['promo_discount'] = 0
            return result

        def calculate_price(make, color, size, year, **kwargs):
            if not kwargs['is_in_stock'] or (make == 'A, Inc.' and year == 2000):
                return 0

            if year == 2020:
                price = 20
            else:
                price = 12

            if size == 'big':
                price += 2

            price -= kwargs['bulk_discount']
            price *= 1 - kwargs['promo_discount']

            return price

        friendly = compute_ndarray(['Make', 'Color', 'Size', 'Year'],
                                   self.makes,
                                   self.colors,
                                   calculate_bulk_discount,
                                   calculate_promo_discount,
                                   self.sizes,
                                   self.years,
                                   calculate_price,
                                   dtype=np.float32)

        self.assertTrue(np.all(friendly.array == self.golden_array))


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.makes = ['A, Inc.', 'B, Inc.']
        self.colors = ['red', 'blue', 'green']
        self.sizes = ['small', 'big']
        self.years = [2000, 2020]
        self.array = np.array(
            [[[[ 0,  1], [ 2,  3]],
              [[ 4,  5], [ 6,  7]],
              [[ 8,  9], [10, 11]]],
             [[[12, 13], [14, 15]],
              [[16, 17], [18, 19]],
              [[20, 21], [22, 23]]]])

        # Axis names are lowercase to support kwargs indexing
        self.friendly = fm.ndarray(self.array,
                                   ['make', 'color', 'size', 'year'],
                                   self.makes, self.colors, self.sizes, self.years)

        # Example formatted string does not contain dim names
        self.friendly_formatted = open(os.path.join('tests', 'friendly_formatted.txt'),
                                       encoding='utf-8').read()

    def test_take_using_dict(self):
        indices_dict = {
            'make': 'A, Inc.',
            'color': ['red', 'blue'],
            'year': [2000, 2020],
        }
        sub_array = self.friendly.take(indices_dict)
        sub_array_expected = self.array[0, :2, :, :2]

        self.assertTrue(np.all(sub_array.array == sub_array_expected))

    def test_take_using_args(self):
        sub_array = self.friendly.take('A, Inc.',
                                       ['red', 'blue'],
                                       None,
                                       [2000, 2020])
        sub_array_expected = self.array[0, :2, :, :2]

        self.assertTrue(np.all(sub_array.array == sub_array_expected))

    def test_take_using_kwargs(self):
        sub_array = self.friendly.take(make='A, Inc.',
                                       year=[2000, 2020],
                                       color=['red', 'blue'])
        sub_array_expected = self.array[0, :2, :, :2]

        self.assertTrue(np.all(sub_array.array == sub_array_expected))

    def test_take_using_call(self):
        indices_dict = {
            'make': 'A, Inc.',
            'color': ['red', 'blue'],
            'year': [2000, 2020],
        }
        sub_array = self.friendly(indices_dict)
        sub_array_expected = self.array[0, :2, :, :2]

        self.assertTrue(np.all(sub_array.array == sub_array_expected))

    def test_get_using_args(self):
        indices = ('A, Inc.', 'green', 'big', 2000)
        value = self.friendly.get(*indices)
        value_expected = self.array[0, 2, 1, 0]

        self.assertEqual(value, value_expected)

    def test_get_using_call(self):
        indices = ('A, Inc.', 'green', 'big', 2000)
        value = self.friendly(*indices)
        value_expected = self.array[0, 2, 1, 0]

        self.assertEqual(value, value_expected)

    def test_set(self):
        value_expected = 999
        indices = ('A, Inc.', 'green', 'big', 2000)
        self.friendly.set(value_expected, *indices)
        value = self.array[0, 2, 1, 0]
        
        self.assertEqual(value, value_expected)

    def test_format(self):
        formatted = self.friendly.formatted()

        self.assertEqual(formatted, self.friendly_formatted)

    def test_format_topological_order(self):
        order = ['year', 'size', 'color', 'make']
        axes = [self.years, self.sizes, self.colors, self.makes]
        formatted = self.friendly.formatted(
            topological_order=order,
            formatter=None,
            display_dim_names=False)

        for i, line in enumerate(formatted.split('\n')[:len(axes)]):
            expected_key = axes[i][0]
            self.assertTrue(str(expected_key) in line)

    def test_format_formatter(self):
        formatted = self.friendly.formatted(
            topological_order=None,
            formatter=lambda v: str(v) + '~~~',
            display_dim_names=False)

        self.assertTrue('18~~~' in formatted)

    def test_format_display_dim_names(self):
        formatted = self.friendly.formatted(
            topological_order=None,
            formatter=None,
            display_dim_names=True)

        self.assertTrue('make = A, Inc.:' in formatted)

    def test_from_formatted(self):
        from_formatted = fm.from_formatted(self.friendly_formatted)

        for arr1, arr2 in zip(from_formatted.dim_arrays, self.friendly.dim_arrays):
            self.assertTrue(np.all(np.array(arr1, dtype=np.str)
                                   == np.array(arr2, dtype=np.str)))
        self.assertTrue(np.all(np.array(from_formatted.array, dtype=np.str)
                               == np.array(self.friendly.array, dtype=np.str)))


class TestNumpyOperations(unittest.TestCase):
    def setUp(self):
        self.x1 = ['0', '1']
        self.y1 = ['0', '1']
        self.array1 = np.array([[0, 1], [2, 3]])
        self.friendly1 = fm.ndarray(self.array1,
                                    ['x', 'y'],
                                    self.x1, self.y1)
        self.x2 = ['2', '3']
        self.y2 = ['2', '3']
        self.array2 = np.array([[0.0, 0.1], [0.2, 0.3]])
        self.friendly2 = fm.ndarray(self.array2,
                                    ['x', 'y'],
                                    self.x2, self.y2)


    def test_swapaxes(self):
        result = fm.swapaxes(self.friendly1, 'x', 'y')
        friendly_expected = fm.ndarray(np.swapaxes(self.array1, 0, 1),
                                       ['y', 'x'],
                                       self.y1, self.x1)

        self.assertEqual(result, friendly_expected)

    def test_moveaxis(self):
        result = fm.moveaxis(self.friendly1, 'x', 'y')
        friendly_expected = fm.ndarray(np.moveaxis(self.array1, 0, 1),
                                       ['y', 'x'],
                                       self.y1, self.x1)

        self.assertEqual(result, friendly_expected)

    def test_concatenate(self):
        result = fm.concatenate((self.friendly1, self.friendly2),
                                axis=0)
        self.assertEqual(result.shape, (4, 2))

    def test_stack(self):
        result = fm.stack((self.friendly1, self.friendly2),
                                'z', ['5', '6'])
        self.assertEqual(result.shape, (2, 2, 2))

    def test_flip(self):
        result = fm.flip(self.friendly1)
        friendly_expected = fm.ndarray(np.flip(self.array1),
                                       ['x', 'y'],
                                       list(reversed(self.x1)), list(reversed(self.y1)))
        self.assertEqual(result, friendly_expected)

    def test_aggregate(self):
        result = self.friendly1.mean(axis=0)
        friendly_expected = fm.ndarray(self.array1.mean(axis=0),
                                       ['y'],
                                       self.y1)


if __name__ == '__main__':
    unittest.main()