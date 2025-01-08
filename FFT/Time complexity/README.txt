Time complexity tests for polynomial-aimed FFT implementation. Points are
fitted with a linear function, presenting O(n) complexity.
Discontinuities may be result of how algorithm is implemented: calculating for
a d-degree polynomial n >= d points, with n a power of two.
This hypothesis is tested seeing whether discontinuities disappear for
d = powers of two.
(x-axis is d and y-axis is seconds taken. I'll add this to latter graphs.)