import numpy as np

from scipy.interpolate import splev
from matplotlib import pyplot as plt


class BSplineBasis:

    def __init__(self, full_knots, degree):
        self._knots = np.array(full_knots)
        self._degree = int(degree)
        self._dimension = len(self._knots) - self._degree - 1
        self._tck = (self._knots, np.eye(self._dimension), self._degree)

    def __len__(self):
        return self._dimension

    def __call__(self, x):
        Bt = np.array(splev(x, self._tck))
        return Bt.T

    def __repr__(self):
        knot_str = '[' + ', '.join(str(k) for k in self._knots) + ']'
        return 'BSplineBasis({}, {})'.format(knot_str, self._degree)

    def plot(self, grid_size=200):
        x = np.linspace(self._knots[0], self._knots[-1], grid_size)
        for y in self(x).T:
            plt.plot(x, y)

    @classmethod
    def uniform(cls, low, high, num_bases, degree):
        '''Construct a uniform basis between low and high.

        low : Lower bound of the basis.
        high : Upper bound.
        num_bases : The number of bases (dimension) of the basis.
        degree : The degree of the polynomial pieces.

        returns : A new BSplineBasis.

        '''
        num_knots = num_bases + degree + 1
        num_interior_knots = num_knots - 2 * (degree + 1)
        knots = np.linspace(low, high, num_interior_knots + 2)
        return cls.with_knots(knots, degree)

    @classmethod
    def with_knots(cls, knots, degree):
        '''Construct a basis based on a knot sequence.

        The knot sequence should contain the lower and upper bound as
        the first and last elements of the sequence. The elements in
        between are the interior knots of the basis.

        knots : The knots (including the boundaries and interior knots).
        degree : The degree of the polynomial pieces.

        returns : A new BSplineBasis.

        '''
        knots = list(knots)
        knots = ([knots[0]] * degree) + knots + ([knots[-1]] * degree)
        return cls(knots, degree)


def pspline_penalty(basis, order=1):
    '''Return a differences penalty matrix.

    '''
    D = np.diff(np.eye(len(basis)), order)
    return np.dot(D, D.T)
