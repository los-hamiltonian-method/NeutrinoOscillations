from uncertainties import ufloat
import numpy as np

wl1 = 588.9950 # nm
wl2 = 589.5924 # nm
wl = np.mean([wl1, wl2])

# Calibración
r1 = 5.670
r2 = 5.262
r1_error = r1 * 0.05
r2_error = r2 * 0.05

r1 = ufloat(r1, r1_error)
r2 = ufloat(r2, r2_error)
r = (r1 + r2) / 2

conversion = 1E6
micrometer_error = 1E-2 # mm

d1 = ufloat(14.71, micrometer_error) # mm
d2 = ufloat(16.6, micrometer_error) # mm
print(d2 - d1)

d1 *= conversion # nm
d2 *= conversion # nm

d = (d2 - d1) / r
wl_diff = wl**2 / (2 * d)

print(wl_diff)
