# %%


# 100 numpy exercises

# This is a collection of exercises that have been collected in the numpy mailing list, on stack
# overflow
# and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
# and new
# users but also to provide a set of exercises for those who teach.


# If you find an error or think you've a better way to
#  solve some of them, feel
# free to open an issue at <https://github.com/rougier/numpy-100>.
# File automatically generated. See the documentation to update questions/answers/hints programmatically.

#### 1. Import the numpy package under the name `np` (★☆☆)

import numpy as np

# %%
#### 2. Print the numpy version and the configuration (★☆☆)
np.version.version
np.show_config()
# %%
#### 3. Create a null vector of size 10 (★☆☆)
np.empty([10])
# %%
#### 4. How to find the memory size of any array (★☆☆)
np.empty([10]).nbytes
# %%
#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
help(np.add)
# %%
#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
a = np.empty([10])
a[4] = 1
a
# %%
#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)
b = np.arange(10, 49 + 1)
# %%
#### 8. Reverse a vector (first element becomes last) (★☆☆)
np.flip(b)
# %%
#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
np.arange(0, 9).reshape(3, 3)
# %%
#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
np.array([1, 2, 0, 0, 4, 0]).nonzero()

# %%
#### 11. Create a 3x3 identity matrix (★☆☆)
np.identity(3)
# %%
#### 12. Create a 3x3x3 array with random values (★☆☆)
rng = np.random.default_rng()
rng.random([3, 3, 3])
# %%
#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
d = rng.random([10, 10])
print(d.min())
print(d.max())
# %%
#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
rng.random([30]).mean()
# %%
#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
e = np.ones([5, 5])
e[1:-1, 1:-1] = 0
e
# %%
#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
n = 3
f = rng.random([n, n])
print(f)
e = np.pad(f, pad_width=1, mode="constant", constant_values=0)
e
# %%
#### 17. What is the result of the following expression? (★☆☆)
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1

# %%
#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
np.diag([1, 2, 3, 4], -1)
# %%
#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
np.indices([8, 8]).sum(0) % 2
# or
a = np.zeros((8, 8))
a[::2, ::2] = 1
a[1::2, 1::2] = 1
a
# %%
#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
np.unravel_index(100, [6, 7, 8])
# %%
#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
np.tile([[0, 1], [1, 0]], [4, 4])
# %%
#### 22. Normalize a 5x5 random matrix (★☆☆)
Z = rng.random([5, 5])
print(Z / np.linalg.norm(Z))
# or
Z = (Z - np.mean(Z)) / (np.std(Z))
print(Z)
# %%
#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
color = np.dtype([("r", np.ubyte), ("g", np.ubyte), ("b", np.ubyte), ("a", np.ubyte)])
# %%
#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
print(np.dot(np.ones([5, 3]), np.ones([3, 2])))
print(np.ones([5, 3]) @ np.ones([3, 2]))
# %%
#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
def neg25(A):
    return np.where((A > 3) & (A < 8), -A, A)


neg25(np.arange(10))
# or
Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
# %%
#### 26. What is the output of the following script? (★☆☆)


# %%
#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
Z = np.arange(5)
Z**Z
2 << Z >> 2
Z < -Z
1j * Z
Z / 1 / 1
# Z<Z>Z
# %%
#### 28. What are the result of the following expressions? (★☆☆)
# np.array(0) / np.array(0) - error
# np.array(0) // np.array(0) - error
# np.array([np.nan]).astype(int).astype(float) - makes a (random?) int and then a float
# %%
#### 29. How to round away from zero a float array ? (★☆☆)
def round_away(x):
    a = np.abs(x)
    b = np.floor(a) + np.floor(2 * (a % 1))
    return np.sign(x) * b


# or

Z = np.random.uniform(-10, +10, 10)
print(np.copysign(np.ceil(np.abs(Z)), Z))

# More readable but less efficient
print(np.where(Z > 0, np.ceil(Z), np.floor(Z)))


# %%
#### 30. How to find common values between two arrays? (★☆☆)
Z1 = np.random.randint(0, 10, 10)
Z2 = np.random.randint(0, 10, 10)
print(np.intersect1d(Z1, Z2))
# %%
#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
defaults = np.seterr(all="ignore")
# back to normal
_ = np.seterr(**defaults)
# %%
#### 32. Is the following expressions true? (★☆☆)
# np.sqrt(-1) == np.emath.sqrt(-1)
# guess not
# %%
#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

np.datetime64("today", "D") - np.timedelta64(1, "D")
np.datetime64("today", "D")
np.datetime64("today", "D") + np.timedelta64(1, "D")

# %%
#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
np.arange("2016-07", "2016-08", dtype="datetime64[D]")
# %%
#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
A = np.ones(3) * 1
B = np.ones(3) * 2
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)

# %%
#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
g = rng.random(5) * 10
print(Z - Z % 1)
print(np.floor(g))
print(np.trunc(g))
print(g // 1)
print(g.astype(int))

# %%
#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
np.zeros((5, 5)) + np.arange(5)
# %%
#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
def gen():
    for i in range(10):
        yield i


np.fromiter(gen(), dtype=int)
# %%
#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
np.linspace(0, 1, 11, endpoint=False)[1:]
# %%
#### 40. Create a random vector of size 10 and sort it (★★☆)
rng.random(10).sort()
# %%
#### 41. How to sum a small array faster than np.sum? (★★☆)
np.add.reduce(np.ones(5))
# %%
#### 42. Consider two random array A and B, check if they are equal (★★☆)
np.allclose(rng.random(5), rng.random(5))
np.array_equal(rng.random(5), rng.random(5))
# %%
#### 43. Make an array immutable (read-only) (★★☆)
Z = rng.random(3)
Z.flags.writeable = False
Z[2] = 7
# %%
#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
Z = rng.random((10, 2))
print(Z)
X, Y = Z[:, 0], Z[:, 1]
X = np.sqrt(X**2 + Y**2)
Y = np.arctan2(Y, X)
print([X, Y])
# %%
#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
Z = rng.random(10)
Z[Z.argmax()] = 0
Z

# %%
#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)
Z = np.zeros((5, 5), [("x", float), ("y", float)])
Z["x"], Z["y"] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing="xy")
Z
# %%
#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)
X, Y = rng.random(5), rng.random(5)
C = 1 / np.subtract.outer(X, Y)
print(C)
# %%
#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)
# %%
#### 49. How to print all the values of an array? (★★☆)
print(np.zeros((40, 40)))
np.set_printoptions(threshold=float("inf"))
print(np.zeros((40, 40)))
# %%
#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
Z = rng.random(5)
print(Z)
s = 0.3
print(Z[np.argmin(np.abs(Z - s))])
# %%
#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
# pos_color = np.dtype([('position', [("x", np.float64), ("y", np.float64)]), ('color': [("r", np.uint), ("g", np.uint), ("b", np.uint), ("a", np.uint)])])

np.zeros(
    10,
    [
        ("position", [("x", float), ("y", float)]),
        ("color", [("r", float), ("g", float), ("b", float)]),
    ],
)

# print(pos_color)
# %%
#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
Z = rng.random((100, 2))
X, Y = np.atleast_2d(Z[:, 0], Z[:, 0])
distances = np.sqrt(X - (X.T) ** 2 + Y - (Y.T) ** 2)
# %%
#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
rng.random(5, dtype=np.float32).view(dtype=np.int32)

# %%
#### 54. How to read the following file? (★★☆)
from io import StringIO

s = StringIO(
    """1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11"""
)
np.genfromtxt(s, delimiter=",")

# %%
#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
list(np.ndenumerate(rng.random((5, 3))))
# %%
#### 56. Generate a generic 2D Gaussian-like array (★★☆)
from scipy.stats import norm

norm.stats(moments="mvsk")
norm.ppf(0.01)
norm.pdf(3)

norm.pdf(np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10)))

# %%
#### 57. How to randomly place p elements in a 2D array? (★★☆)

# %%
#### 58. Subtract the mean of each row of a matrix (★★☆)
Z = rng.random((2, 2))
Z
print(Z)
print(Z - Z.mean(axis=1, keepdims=True))
# %%
#### 59. How to sort an array by the nth column? (★★☆)
Z = rng.random((4, 4))
print(Z)
n = 2
row_indices = np.argsort(Z[:, n])
Z[row_indices]

# %%
#### 60. How to tell if a given 2D array has null columns? (★★☆)
Z = np.zeros((2, 2))
~np.any(Z, axis=0).any()
# %%
#### 61. Find the nearest value from a given value in an array (★★☆)
Z = rng.random((2, 2))
v = 0.3
Z.flat[np.abs(Z - v).argmin()]
# %%
#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
A = np.arange(3).reshape(3, 1)
B = np.arange(3).reshape(1, 3)
it = np.nditer([A, B, None])
for x, y, z in it:
    z[...] = x + y
print(it.operands[2])
# %%
#### 63. Create an array class that has a name attribute (★★☆)

# %%
#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
A = np.arange(9)
# B=np.random.randint(9)
B = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
A += np.bincount(B, minlength=A.size)
A

# or
np.add.at(Z, I, 1)
print(Z)
# %%
#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

# %%
#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
img = np.random.randint(2, size=(4, 4, 3))
np.unique(img.reshape(-1, 3), axis=0).size
# %%
#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
Z = np.random.randint(5, size=(2, 3, 4, 5))
Z.reshape(2, 3, -1).sum(axis=-1)
# %%
#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)
D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
import pandas as pd

print(pd.Series(D).groupby(S).mean())

# %%
#### 69. How to get the diagonal of a dot product? (★★★)

# %%
#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
A = np.array([1, 2, 3, 4, 5])
Z = np.zeros(A.size * 4)
Z[::4] = A
Z
# %%
#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
# %%
#### 72. How to swap two rows of an array? (★★★)
# %%
#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)
# %%
#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

#### 75. How to compute averages using a sliding window over an array? (★★★)

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)

#### 82. Compute a matrix rank (★★★)

#### 83. How to find the most frequent value in an array?

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

#### 88. How to implement the Game of Life using numpy arrays? (★★★)

#### 89. How to get the n largest values of an array (★★★)

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

#### 91. How to create a record array from a regular array? (★★★)

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)

#### 95. Convert a vector of ints into a matrix binary representation (★★★)

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
