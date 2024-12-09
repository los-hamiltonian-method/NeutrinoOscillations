import cv2 as cv
import numpy as np

filename = './gaussian_pulses/gaussian_pulse_low_dispersion_4'
extension = '.mp4'
fps = 33

cap = cv.VideoCapture(filename + extension)
dimensions = np.uint16((cap.get(3), cap.get(4)))
output = cv.VideoWriter(f"{filename}_fps{fps}{extension}",
	cv.VideoWriter_fourcc(*'mp4v'), fps, dimensions)

if cap.isOpened():
	success, frame = cap.read()
else:
	sucess = False

while success:
	success, frame = cap.read()

	if not success:
		break

	output.write(frame)

output.release()
cap.release()

