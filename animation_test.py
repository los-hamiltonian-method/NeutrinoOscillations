import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation as FA
from matplotlib.animation import FFMpegWriter
import numpy as np

mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'

k = 0.5
w = 0.05

@np.vectorize
def func(x, t):
	return np.cos(k * x - w * t)


x = np.linspace(-10 * np.pi, 10 * np.pi, 1000)

fig, ax = plt.subplots()
ax.set(xlim=(-10*np.pi, 10*np.pi))

def grapher(t):
	ax.clear()
	ax.set(title=t)
	ax.plot(x, func(x, t), color='black')


filename = './animation.mp4'
animation = FA(fig, grapher, frames=(list(range(600))), repeat=False)
writer = FFMpegWriter(fps=60)
animation.save(filename, writer=writer)
