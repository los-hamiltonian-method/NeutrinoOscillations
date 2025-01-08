#%% Imports
import numpy as np
from time import time
from typing import Callable


#%% MiniPies
'''
Functions implemented within this code can be imported to a new Python file
and run without any further importing. However, if you'd like to run main()
from this .py you'll need to import the MiniPys from:
github.com/emilio-moreno/MiniPys
'''

minipy_formatter_path = r'C:\Users\Laevateinn\Documents\GitHub' \
                        r'\MiniPys\Formatter'

minipy_tictac_path = r'C:\Users\Laevateinn\Documents\GitHub' \
                     r'\MiniPys\tictac'


#%% FFT implementation for Polynomial
class Polynomial(object):
    
    @staticmethod
    def vectorize(x: float, degree: int):
        '''Builds ndarray [[1], [x], ..., [x^degree]].'''
        X = []
        for k in range(degree + 1):
            X.append([x**k])
        return np.array(X)

    @staticmethod
    def evaluate(x: float, coefficients: np.ndarray):
        '''Evaluates polynomial at x.'''
        degree = len(coefficients[:, 0]) - 1
        eva = np.transpose(Polynomial.vectorize(x, degree)) @ coefficients
        return eva[0, 0]
    
    # Think of a better name for this
    def eva(self, x: float):
        '''Evaluates Polynomial at x.'''
        return self.evaluate(x, self.coefficients)
    
    @staticmethod
    def extract(coefficients: np.array):
        '''Extracts even and odd coefficients.'''
        degree = int(len(coefficients[:, 0]) - 1)
        filter_array = [bool((k + 1) % 2) for k in range(degree + 1)]
        
        evens = coefficients[:, 0][filter_array]
        odds = coefficients[:, 0][np.invert(filter_array)]
        evens = np.reshape(evens, (len(evens), 1))
        odds = np.reshape(odds, (len(odds), 1))
        
        return evens, odds
    
    def __init__(self, coefficients: np.ndarray):
        '''
        Instantiate polynomial. Coefficients is a (d, 1) representation
        of polynomial coefficients. With index corresponding to the
        coefficients associated degree.
        '''
        if not coefficients.shape == (len(coefficients[:, 0]), 1):
            raise ValueError("Array must have shape (d, 1). "
                            f"Received {coefficients.shape}.")
            
        self.coefficients = coefficients
        self.degree = int(len(coefficients[:, 0]) - 1)
        self.evens, self.odds = self.extract(coefficients)
    
    def FFT(self):
        '''FFT method for Polynomial class.'''
        return Polynomial.fft(self.coefficients)
    
    @staticmethod
    def fft(coefficients, base=None):
        '''
        Calculates polynomial values at n >= degree of polynomial
        roots of unity, with n a power of 2. Returns a (n, 1) ndarray.
        Works for any degree polynomial.
        '''
        n = len(coefficients)
        if n == 1:
            return coefficients
        
        evens, odds = Polynomial.extract(coefficients)
        # We won't provide a base when starting the algorithm, to allow for
        # padding.
        if not base:
            # There may be a better way to calculate n.
            n = int(2**np.ceil(np.log2(n)))
            base = np.exp(1j * 2 * np.pi / n)
            even_padding = np.zeros((int(n / 2) - len(evens), 1))
            odd_padding = np.zeros((int(n / 2) - len(odds), 1))
            evens = np.append(evens, even_padding, axis=0)
            odds = np.append(odds, odd_padding, axis=0)
            
        result = np.zeros((n, 1), dtype='complex_')
    
        y_e = Polynomial.fft(evens, base**2)
        y_o = Polynomial.fft(odds, base**2)
        
        for k in range(int(n / 2)):
            result[k, :] = y_e[k, :] + base**k * y_o[k, :]
            result[int(k + n / 2)] = y_e[k, :] - base**k * y_o[k, :]
        return result

    def __repr__(self):
        return f'''Polynomial={self.coefficients}'''
    
    def __str__(self):
        str_p = "$"
        for k, c in enumerate(self.coefficients[:-1, 0]):
            if c == 0:
                continue
            if c == 1:
                c = ""
            str_p += f"{c} x^{k} + "
        
        str_p += f"{self.coefficients[-1, 0]} x^{self.degree}$"
        return str_p


#%% main()
def main():
    # Imports
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, minipy_formatter_path)
    import minipy_formatter as MF

    # Colors
    # CMU = r'C:\Users\Laevateinn\AppData\Local\Microsoft\Windows\Fonts' \
    #      r'\cmunrm.ttf'
    # MF.Format([CMU]).rcUpdate()
    MF.Format().rcUpdate()
    colors = MF.Colors(('miku', '#5dd4d8'), ('rin', '#ecd38c'),
                    ('darkrin', '#c6b176'), ('lgray', '#c0c0c0'))
    
    # FFT test
    n = int(np.random.uniform(0, 1000, 1)[0])
    coeff = np.random.uniform(-1000, 1000, (n, 1))
    P = Polynomial(coeff)
    result = P.FFT()
    
    n = int(2**np.ceil(np.log2(n)))
    base = np.exp(1j * 2 * np.pi / n)
    
    validation = []
    for k in range(n):
        validation.append([P.eva(base**k)])
    validation = np.array(validation)
    
    fig, ax = plt.subplots()
    ax.scatter(result, validation, color=colors.miku.hex)
    ax.set(title="Validation", xlabel="FFT evaluation",
           ylabel="Numpy evaluation", aspect='equal')
    ax.grid(linestyle='--', color=colors.lgray.hex)
    plt.show()


    # Complexity tests    
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    sys.path.insert(0, minipy_tictac_path)
    import tictac as tt
    
    # Code to be tested
    def precode(i):
        coeff = np.random.uniform(-100, 100, (i, 1))
        global P
        P = Polynomial(coeff)

    def code(i):
        global P
        P.FFT()
    
    # Test params    
    upper_bound = int(2**14 + 3)
    #upper_bound = int(2**12 + 3)
    lower_bound = 2
    step = 100
    n, times_avg = tt.tictac(lower_bound, upper_bound, step,
                             precode=precode, code=code, total_tests=3)
    tt.tictac()
    
    # Trying a linear fit
    f = lambda n, a: a * n
    opt, cov = curve_fit(f, n, times_avg)
    
    N = np.array([lower_bound, upper_bound])
    fig, ax = plt.subplots()
    ax.plot(N, f(N, opt[0]),
            label=f"$\mathcal{{O}}(n)$  Variance $= {cov[0, 0]:.3E}$",
            color=colors.darkrin.hex)
    ax.scatter(n, times_avg, color=colors.rin.hex)
    
    ax.set(title="Time complexity tests for polynomial-aimed FFT",
           xlabel="Polynomial degree ($d$)", ylabel="Time taken (s)")
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', color=colors.lgray.hex)
    plt.show()


#%% 呪い
if __name__ == '__main__':
    main()