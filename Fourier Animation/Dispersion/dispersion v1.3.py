import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from matplotlib.animation import FFMpegWriter
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from time import time
from playsound import playsound

# rcParams
# mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'
mpl.rcParams['animation.ffmpeg_path'] = r'E:\Program Files\ffmpeg-7.1-full_build\bin\ffmpeg.exe'

rc_update = {'font.size': 10, 'font.family': 'serif',
			 'font.serif': ['Times New Roman', 'FreeSerif'], 'mathtext.fontset': 'cm'}
plt.rcParams.update(rc_update)

og_colors = {'blue': '#18CCF5', 'red': '#FA3719'}
colors = {'orange': '#ffab40', 'Lgrey': '#999', 'blue': '#7ea8a7',
		  'red': '#e58e8fff', 'yellow': '#fcf3cc', 'dark_orange': '#bf6f22'}

class FourierFunction(object):

	@staticmethod
	def show_plot():
		'''Graphs plot full screen.'''
		figManager = plt.get_current_fig_manager()
		figManager.full_screen_toggle()
		plt.tight_layout()
		plt.show()


	@staticmethod
	def animation_progress(total_frames, frame):
		'''Prints progress of animation.'''
		# Progress bar str
		frame += 1
		done = frame / total_frames
		remaining = (total_frames - frame) / total_frames
		done = int(np.ceil(20 * done)) * '#'
		remaining = int(np.ceil(20 * remaining)) * '_'
		progress_bar = "[" + done + remaining + "]"

		# Print
		print(f"Animated: {frame} / {total_frames} frames")
		print(progress_bar + '\n')

		if frame == total_frames:
			print("Animation done!\n")


	def __init__(self, name, func, A_terms=lambda k: 0, B_terms=lambda k: 0, w=lambda k: 0):
		'''
		FourierFunction object.

		Parameters
		----------
		func: function
			Function to be Fourier expanded.
		A_terms: function
			Defaults to 0. Fourier transform for cosine terms.
		B_terms: function
			Defaults to 0. Fourier transform for sine terms.
		w: function
			Defaults to 0. Function of k, dispersion relation.
		'''
		self.name = name
		self.func = func
		self.w = w

		self.A_terms = A_terms
		self.B_terms = B_terms
		self.phase = lambda k, x, t: k * x - w(k) * t
		self.A_cos = lambda k, x, t: A_terms(k) * np.cos(self.phase(k, x, t))
		self.B_sin = lambda k, x, t: B_terms(k) * np.sin(self.phase(k, x, t))


	def FIntegrate(self, x, k, t=0, cutoff_ratio=0.01):
		'''
		Returns Fourier integral for self.
		
		Parameters
		----------
		x: array-like
			x-space values.
		k: array-like
			k-space values.
		t: float
			Defaults to 0. Time value.
		cutoff_ratio: float
			Defaults to 0.01. Coefficientes below this ratio of
			the maximum are discarded.
		
		Returns
		-------
		FIntegral(x, t): array-like
			Fourier integral of x values evaluated at time t.
		'''
		even_integrand = lambda k, x, t: self.A_cos(k, x, t)
		odd_integrand = lambda k, x, t: self.B_sin(k, x, t)

		relevant_A_k, relevant_B_k = self.get_relevant(k, cutoff_ratio=cutoff_ratio)

		even_limits = relevant_A_k[0], relevant_A_k[-1]
		odd_limits = relevant_B_k[0], relevant_B_k[-1]

		@np.vectorize
		def FIntegral(x, t):
			even_integral = lambda x, t: sp.integrate.quad(
				even_integrand, even_limits[0], even_limits[1], args=(x, t))
			odd_integral = lambda x, t: sp.integrate.quad(
				odd_integrand, odd_limits[0], odd_limits[1], args=(x, t))
			return even_integral(x, t) + odd_integral(x, t)

		return FIntegral(x, t)[0] / np.sqrt((2 * np.pi))


	def get_relevant(self, k, cutoff_ratio):
		'''
		Gets relevant k-valeus.
		A single k-value is relevant if the absolute value of A_terms(k) (or
		B_terms(k)) is above the maximum value of the absolute values in
		A_terms (or B_terms) times the cut off ratio.
		'''


		relevant_A_k = []
		relevant_B_k = []

		# This could probably be done with Pandas
		A_cutoff = cutoff_ratio * max(np.abs(self.A_terms(k)))
		B_cutoff = cutoff_ratio * max(np.abs(self.B_terms(k)))
		for i in k:
			if abs(self.A_terms(i)) > A_cutoff: relevant_A_k.append(i)
			if abs(self.B_terms(i)) > B_cutoff: relevant_B_k.append(i)

		if not relevant_A_k: relevant_A_k = [0]
		if not relevant_B_k: relevant_B_k = [0]

		relevant_A_k = np.array(relevant_A_k)
		relevant_B_k = np.array(relevant_B_k)
		return relevant_A_k, relevant_B_k


	def graph_relevant(self, axs, k, cutoff_ratio=1 / (2 * np.e), wk_relation=False):
		'''
		Graphs relevant range of values for k-space (and optionally for wk-relation).

		Parameters
		----------
		axs: dict(matplotlib.axis)
			Dictionary containing axis. Must include axis named 'k-space' (and optionally
			axis named 'wk-relation').
		k: array-like
			k-space values.
		cutoff_ratio: float
			Defaults to 1 / 2e. Cut off ratio.
		wk_relation: bool
			Defaults to False. Whether or not to plot relevant values for wk relation.
		'''
		relevant_A_k, relevant_B_k = self.get_relevant(k, cutoff_ratio)
		axs['k-space'].plot(relevant_A_k, self.A_terms(relevant_A_k),
							color=og_colors['red'], label='Relevant $A(k)$')
		axs['k-space'].plot(relevant_B_k, self.B_terms(relevant_B_k),
		  				    color=og_colors['blue'], label='Relevant $B(k)$')

		# wk-relation
		if not wk_relation:
			return

		# Axis cleared for animating
		axs['wk-relation'].clear()
		relevant_k = relevant_A_k if len(relevant_A_k) > len(relevant_B_k) else relevant_B_k
		axs['wk-relation'].plot(relevant_k, self.w(relevant_k),
								color=colors['orange'], label='Relevant $k$', zorder=1)


	def graph(self, fig, axs, x, k, t=0, k_space=False, wk_relation=False,
			  plot_original=False, show=False, animating=False):
		'''
		Graphs fourier integral of self.func. Optionally, the original function,
		k-space and wk relation can be plotted.

		Parameters
		----------
		fig: matplotlib.figure
			Figure in which to plot fourier integral.
		axs: dict(matplotlib.axis)
			Dictionary containing axis.
		x: array-like
			x-space values.
		k: array-like
			k-space values in which to expand self.func. Values below a threshold are
			ignored.
		t: float
			Defaults to 0. Simulation time value. May be slowed down with respect to real time.
		k_space: bool
			Defaults to False. Whether or not to plot k-space.
		wk_relation: bool
			Defaults to False. Whether or not to plot wk relation.
		plot_original: bool
			Defaults to False. Whether or not to plot original function in x-space.
		show: bool
			Defaults to False. Whether or not to show plot.
		animating: bool
			Defaults to False. Whether or not code is animating. If animating, fig, axs
			won't be returned.

		Returns
		-------
		fig: matplotlib.figure
			Original figure.
		axs: dict(matplotlib.axis)
			Original axis dictionary.
		'''

		# Plots
		fig.suptitle(f"Fourier integral of {self.name} "
					 f"on range $k \in [{min(k)}, {max(k)}]$", fontsize=10)
		plt.subplots_adjust(hspace=0.43, wspace=0.35)

		# x-space
		legend_fontsize = 2
		
		# Axis must be cleared when animating
		axs['x-space'].clear()

		# Limits and labels
		x_padding = max(x) / 20
		y_max = max(abs(self.func(x)))
		y_padding = y_max / 10
		y_plt_range = (-max(abs(self.func(x))) - y_padding, max(abs(self.func(x))) + y_padding)
		axs['x-space'].set(ylim=y_plt_range, xlim=(min(x), max(x)))
		axs['x-space'].set(title='$x$-space', xlabel='$x$', ylabel='$E(x)$')

		# Animation time
		if animating:
			axs['x-space'].text(min(x) + x_padding / 2,
							    y_max - y_padding, f"t = {round(t, 3)}s")

		if plot_original:
			axs['x-space'].plot(x, self.func(x), label='Original function',
								color=colors['blue'], zorder=1, linestyle='--')
		axs['x-space'].plot(x, self.FIntegrate(x, k, t),
							label='Fourier integral', color=colors['orange'], zorder=0)
		#axs['x-space'].legend(fontsize=legend_fontsize)

		# k-space
		if not k_space:
			if show: FourierFunction.show_plot()
			if not animating: return fig, axs
			return

		# Axis must be cleared when animating
		axs['k-space'].clear()
		axs['k-space'].set(title='$k$-space', xlabel='$k$', ylabel='$A(k)$/$B(k)$')
		axs['k-space'].plot(k, self.A_terms(k), label='$A(k)$: Cosine terms',
							   color=colors['red'], zorder = 2)
		axs['k-space'].plot(k, self.B_terms(k), label='$B(k)$: Sine terms',
							linestyle='--', color=colors['blue'])
		self.graph_relevant(axs, k, wk_relation=wk_relation)
		axs['k-space'].set(xlim=(min(k), max(k)))
		
		average_k = round(np.average(k), 2)
		axs['k-space'].axvline(average_k, linestyle='-.',
							   linewidth=1, color=colors['orange'])
		new_ticks = list(axs['k-space'].get_xticks()) + [average_k]
		axs['k-space'].set_xticks(new_ticks)
		#axs['k-space'].legend(fontsize=legend_fontsize)

		# wk-relation
		if not wk_relation:
			if show: FourierFunction.show_plot()
			if not animating: return fig, axs
			return

		# Axis must be cleared when animating
		axs['wk-relation'].set(xlim=(0, max(k)))
		axs['wk-relation'].set(title='$\omega k $-relation', xlabel='$k$', ylabel='$\omega(k)$')
		axs['wk-relation'].plot(k, self.w(k), label='$\omega(k)$',
							    color=colors['blue'], zorder=0)
		#axs['wk-relation'].axhline(0, color='#111', ls='--', linewidth=0.5)
		axs['wk-relation'].set(xlim=(min(k), max(k)))
		#axs['wk-relation'].legend(fontsize=legend_fontsize)

		# Adding average frequency
		axs['wk-relation'].axvline(average_k, linestyle='-.', linewidth=1, color=colors['orange'])
		new_ticks = list(axs['wk-relation'].get_xticks()) + [average_k]
		axs['wk-relation'].set_xticks(new_ticks)
		

		if show: FourierFunction.show_plot()
		if not animating: return fig, axs
		return


	def animate(self, filename, x, k, total_frames, fps=10, slowdown=20,
				k_space=False, wk_relation=False, plot_original=False):
		'''
		Animates evolution of wave pulse given initial shape self.func.
		Animation is saved as a video file.

		Parameters
		----------
		filename: str
			File name with which animation will be saved. Must include a
			video extension.
		x: array-like
			x-space values.
		k: array-like
			k-space values.
		total_frames: int
			Total frames to be animated.
		fps: int
			Defaults to 10. Frames per second.
		slowdown: float
			Defaults to 20. Slowdown factor with respect to real time,
			i.e., simulation time = real time / slowdown
		k_space: bool
			Defaults to False. Whether or not to graph k-space.
		wk_relation: bool
			Defaults to False. Whether or not to graph wk relation.
		plot_relation: bool
			Defaults to False. Whether or not to plot function at t=0.
		'''

		subplot_mosaic = [['x-space', 'x-space']]
		if k_space:
			if wk_relation:
				subplot_mosaic.append(['k-space', 'wk-relation'])
			else:
				subplot_mosaic.append(['k-space', 'k-space'])
		fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)

		def create_updater(frame, k_space, wk_relation):
			t = frame / (fps * slowdown)
			# Only need to graph k-space and wk-relation once.
			if t > 0:
				k_space, wk_relation = [False, False]

			self.graph(fig, axs, x, k, k_space=k_space, wk_relation=wk_relation,
					   t=t, animating=True)
			self.animation_progress(total_frames, frame)

		updater = lambda frame: create_updater(frame, k_space, wk_relation)
		animation = FA(plt.gcf(), updater, frames=list(range(total_frames)), repeat=False)
		writer = FFMpegWriter(fps=fps)
		animation.save(filename, writer=writer)


def main():
	# Pulse parameters
	p_i = 25
	m_i = 15
	std = 0.2
	A = 20


	@np.vectorize
	def x_gaussian(x, mean=0, std=1):
		normal = A / (std * np.sqrt(2 * np.pi))
		return normal * np.e**(-(x / std)**2 / 2) * np.cos(p_i * x)

	@np.vectorize
	def p_gaussian(p, carrier_p=0, x_std=1):
		p = p - carrier_p
		B = A / np.sqrt(2 * np.pi)
		return B * np.e**(-(x_std * p)**2 / 2)

	def even_terms(p, carrier_p=0, x_std=1):
		'''Fourier transform for even terms.'''
		return p_gaussian(p, carrier_p, x_std)

	@np.vectorize
	def B_terms(m):
		'''Fourier transform for odd terms.'''
		return 0

	@np.vectorize
	def w(k):
		return np.sqrt(k**2 + m_i**2)

	f = lambda x: x_gaussian(x, std=std)
	A_terms = lambda k: even_terms(k, p_i, std)
	
	# Plot parameters (only for FourierFunction.graph)
	# FourierFunction.animate creates their own.
	subplot_mosaic = [['x-space', 'x-space'], ['k-space', 'wk-relation']]
	fig, axs = plt.subplot_mosaic(subplot_mosaic, dpi=300)
	
	# Space variables
	rang = 25
	x = np.linspace(-10 * std, 50 * std, 1300)
	k = np.linspace(-rang + p_i, rang + p_i, 500)

	# FourierFunction
	gaussian_pulse = FourierFunction("gaussian pulse", f, A_terms, B_terms, w)
	#gaussian_pulse.graph(fig, axs, x, k, plot_original=True,
	#					 show=True, k_space=True, wk_relation=True)
	
	# Need a way to not overwrite old files
	filename = '../mp4/gaussian_pulses/gaussian_pulse_low_dispersion_4.mp4'
	gaussian_pulse.animate(filename, x, k, fps=10, slowdown=10, total_frames=1000,
						   k_space=False, wk_relation=False)
 
	done_sound = '../GW150914_H1_shifted.wav'
	playsound(done_sound)


if __name__ == '__main__':
	main()
