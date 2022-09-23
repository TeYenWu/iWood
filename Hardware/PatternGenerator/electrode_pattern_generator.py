import numpy as np
import math
import glob
# import cv2
import json 
from PIL import Image

screw_diameter = 3/8*25.4 + 1
minimum_separation = screw_diameter * math.sqrt(2)
trace_width = 2
board_size = 610
color = [0, 0, 0, 255]


def generate_png_pattern(width, height, electrode_size, minimum_separation, cropping):
	electrode_diagonal = int(electrode_size * math.sqrt(2))
	# print(electrode_diagonal)
	gap = int(minimum_separation * math.sqrt(2))
	# print(gap)
	top_unit_matrix = np.full((int(electrode_diagonal+gap), int(electrode_diagonal+gap), 4), 0,dtype=np.uint8)
	bot_unit_matrix = np.full((int(electrode_diagonal+gap), int(electrode_diagonal+gap), 4), 0,dtype=np.uint8)
	for i in range(int(electrode_diagonal+gap)):
		for j in range(int(electrode_diagonal+gap)):
			###top layer
			if i < int(electrode_diagonal/2):
				if j < int(electrode_diagonal/2) - i or j > int(electrode_diagonal/2) + gap + i:
					top_unit_matrix[i][j] = color
			if i >= electrode_diagonal/2 + gap:
				if j < electrode_diagonal/2 - (electrode_diagonal + gap - 1 - i) or j > electrode_diagonal/2 + gap +  (electrode_diagonal + gap - 1 - i):
					top_unit_matrix[i][j] = color
			# if i < trace_width or j < trace_width or i > electrode_diagonal+gap-trace_width-1 or j > electrode_diagonal+gap-trace_width-1:
			# 	top_unit_matrix[i][j] = color

			### bottom layer
			if i > gap/2 and i < (gap+electrode_diagonal)/2:
				if j > gap / 2 + electrode_diagonal/2 - (i - gap/2) and j < (electrode_diagonal + gap) - gap/2 - (electrode_diagonal/2 - (i - gap/2)):
					bot_unit_matrix[i][j] = color
			if i < ((gap+electrode_diagonal) - gap/2) and i >= (gap+electrode_diagonal)/2:
				if j >  gap / 2 + (i - (gap+electrode_diagonal)/2) and j < (electrode_diagonal + gap) - gap/2 - (i - (gap+electrode_diagonal)/2):
					bot_unit_matrix[i][j] = color
			# if (i >= int(electrode_diagonal+gap)/ 2 - trace_width/2 and i < int(electrode_diagonal+gap)/ 2 + trace_width/2) or (j <= int(electrode_diagonal+gap)/ 2 + trace_width/2 and j >= int(electrode_diagonal+gap)/ 2 - trace_width/2):
			# 	bot_unit_matrix[i][j] = color
	top_board = np.tile(top_unit_matrix, (int(height/(electrode_diagonal+gap))+1, int(width/(electrode_diagonal+gap))+1, 1))
	bot_board = np.tile(bot_unit_matrix, (int(height/(electrode_diagonal+gap))+1, int(width/(electrode_diagonal+gap))+1, 1))
	if cropping:
		top_board = top_board[:int(height), :int(width)]
		bot_board = bot_board[:int(height), :int(width)]
	im = Image.fromarray(top_board)
	im.save("top_without_connection.png")
	bim = Image.fromarray(bot_board)
	bim.save("bot_without_connection.png")
	 	 
def generate_connection_pattern(width, height, electrode_size, minimum_separation, cropping):
	electrode_diagonal = int(electrode_size * math.sqrt(2))
	# print(electrode_diagonal)
	gap = int(minimum_separation * math.sqrt(2))
	# print(gap)
	top_unit_matrix = np.full((int(electrode_diagonal+gap), int(electrode_diagonal+gap), 4), 0,dtype=np.uint8)
	bot_unit_matrix = np.full((int(electrode_diagonal+gap), int(electrode_diagonal+gap), 4), 0,dtype=np.uint8)
	for i in range(int(electrode_diagonal+gap)):
		for j in range(int(electrode_diagonal+gap)):
			###top layer
			if i < trace_width/2 or i > electrode_diagonal+gap-trace_width/2-1:
				if j > int(electrode_diagonal)/4 and j < int(electrode_diagonal)*3/4 + gap:
					top_unit_matrix[i][j] = color
			if j < trace_width/2 or j > electrode_diagonal+gap-trace_width/2-1:
				if i > electrode_diagonal/4 and i <  electrode_diagonal*3/4+gap:
					top_unit_matrix[i][j] = color

			### bottom layer
			# if i > gap/2 and i < (gap+electrode_diagonal)/2:
			# 	if j > gap / 2 + electrode_diagonal/2 - (i - gap/2) and j < (electrode_diagonal + gap) - gap/2 - (electrode_diagonal/2 - (i - gap/2)):
			# 		bot_unit_matrix[i][j] = color
			# if i < ((gap+electrode_diagonal) - gap/2) and i >= (gap+electrode_diagonal)/2:
			# 	if j >  gap / 2 + (i - (gap+electrode_diagonal)/2) and j < (electrode_diagonal + gap) - gap/2 - (i - (gap+electrode_diagonal)/2):
			# 		bot_unit_matrix[i][j] = color
			if (i >= int(electrode_diagonal+gap)/ 2 - trace_width/2 and i < int(electrode_diagonal+gap)/ 2 + trace_width/2):
				if j <  gap / 2 + electrode_diagonal/4 or  j > (electrode_diagonal + gap) - gap/2 - electrode_diagonal/4:  
					bot_unit_matrix[i][j] = color

			if (j <= int(electrode_diagonal+gap)/ 2 + trace_width/2 and j >= int(electrode_diagonal+gap)/ 2 - trace_width/2):
				if i <  gap / 2 + electrode_diagonal/4 or  i > (electrode_diagonal + gap) - gap/2 - electrode_diagonal/4:  
					bot_unit_matrix[i][j] = color
	top_board = np.tile(top_unit_matrix, (int(height/(electrode_diagonal+gap))+1, int(width/(electrode_diagonal+gap))+1, 1))
	bot_board = np.tile(bot_unit_matrix, (int(height/(electrode_diagonal+gap))+1, int(width/(electrode_diagonal+gap))+1, 1))
	if cropping:
		top_board = top_board[:int(height), :int(width)]
		bot_board = bot_board[:int(height), :int(width)]
	print(top_board.shape)
	im = Image.fromarray(top_board)
	im.save("top_connection.png")
	bim = Image.fromarray(bot_board)
	bim.save("bot_connection.png")

if __name__ == "__main__":
	generate_png_pattern(board_size, board_size, 80, minimum_separation, True)
	generate_connection_pattern(board_size, board_size, 80, minimum_separation, True)
