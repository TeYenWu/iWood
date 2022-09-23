import numpy as np
import math
import glob
# import cv2
import json 
from scipy import ndimage
from scipy import signal

import torch
from torch.nn.functional import conv2d

### 
### #24 screw diameter https://www.fastenermart.com/files/wood-screw-dimensions.html
minimum_separation = screw_diameter = 3/8*25.4 + 1
maximum_electrode_size = int(1524 - (minimum_separation*math.sqrt(2))/math.sqrt(2))
# minimum_trace_width = range() ## resistance?
# minimum_separation = screw_diameter = 10

### simulation parameter
### size paramenter from Homedepot
### plywood common size 4 x 8 foot and 5 x 5 foot with 1/2-inch thickness (https://www.homedepot.com/c/ab/types-of-plywood/9ba683603be9fa5395fab909d37f448) 
### lumber size 2 x 4 or 4 x 4 foot (https://www.homedepot.com/c/ab/types-of-lumber/9ba683603be9fa5395fab90567851db)
# simulation_board_sizes = [(1219.2, 2438.4), (1524, 1524), (101.6, 7300)] 
simulation_board_sizes = [(1219.2, 2438.4), (1524, 1524)] 
simulation_max_uniform_sclae = 10
simulation_max_x_sclae = 10
# simulation_board_sizes = [(1000, 1000)] 
simulation_electrode_sizes = range(10, maximum_electrode_size, 10)

### load shape

# test_shapes = load_shape()
shape_scale_precision = 1
shape_rotation_precision = 22.5
shape_position_precision = 10
simulation_shape_rotations =[i*shape_rotation_precision for i in range(int(360/shape_rotation_precision))]

def load_shapes():
	shapes = []
	for filename in glob.glob('./test_shapes/*.csv'):
		matrix = np.loadtxt(filename, delimiter=',')
		print(filename)
		shapes.append((filename, matrix))
	return shapes

def generate_pattern(width, height, electrode_size, minimum_separation):
	electrode_diagonal = int(electrode_size * math.sqrt(2))
	# print(electrode_diagonal)
	gap = int(minimum_separation * math.sqrt(2))
	# print(gap)
	unit_matrix = np.zeros((int(electrode_diagonal+gap), int(electrode_diagonal+gap)))
	for i in range(int(electrode_diagonal+gap)):
		for j in range(int(electrode_diagonal+gap)):
			###top layer
			if i < int(electrode_diagonal/2):
				if j < int(electrode_diagonal/2) - i or j > int(electrode_diagonal/2) + gap + i:
					unit_matrix[i][j] = 1
			if i >= electrode_diagonal/2 + gap:
				if j < electrode_diagonal/2 - (electrode_diagonal + gap - 1 - i) or j > electrode_diagonal/2 + gap +  (electrode_diagonal + gap - 1 - i):
					unit_matrix[i][j] = 1

			### bottom layer
			if i > gap/2 and i < (gap+electrode_diagonal)/2:
				if j > gap / 2 + electrode_diagonal/2 - (i - gap/2) and j < (electrode_diagonal + gap) - gap/2 - (electrode_diagonal/2 - (i - gap/2)):
					unit_matrix[i][j] = -1
			if i < ((gap+electrode_diagonal) - gap/2) and i >= (gap+electrode_diagonal)/2:
				if j >  gap / 2 + (i - (gap+electrode_diagonal)/2) and j < (electrode_diagonal + gap) - gap/2 - (i - (gap+electrode_diagonal)/2):
					unit_matrix[i][j] = -1
	
	board = np.tile(unit_matrix, (int(height/(electrode_diagonal+gap)), int(width/(electrode_diagonal+gap))))
	return np.resize(board, (int(height), int(width)))
	#  np.savetxt("test.csv", board, delimiter = ',', fmt='%1u')

# def conv2d(inputs, kernel, strides):
	# feature_map = np.zeros((
	#     int((inputs.shape[0] - kernel.shape[0] + 1)/strides[0]),
	#     int((inputs.shape[1] - kernel.shape[1] + 1)/strides[1]),
	# ))
	# for x in range(0, inputs.shape[0] - kernel.shape[0] + 1, strides[0]):
	#     for y in range(0, inputs.shape[1] - kernel.shape[1] + 1, strides[1]):
	#         for i in range(kernel.shape[0]):
	#             for j in range(kernel.shape[1]):
	#                 feature_map[int(x/strides[0])][int(y/strides[1])] += inputs[x + i][y + j] * kernel[i][j]
	#         print(feature_map[int(x/strides[0])][int(y/strides[1])])

def main():
	result = {}
	test_shapes = load_shapes()
	for electrode_size in simulation_electrode_sizes:
		result[electrode_size] = {}
		result[electrode_size]["overall_scores"] = []
		result[electrode_size]["simulation_records"] = []
		for simulation_board_size in simulation_board_sizes:
			print("simulation_board_size")
			print(simulation_board_size)
			
			width = simulation_board_size[0]
			height = simulation_board_size[1]
			# electrode_size = (min(width, height)/n - (minimum_separation*math.sqrt(2)))/math.sqrt(2)
			if electrode_size < (min(width, height) - (minimum_separation*math.sqrt(2)))/math.sqrt(2):
				n = int(min(width, height)/(minimum_separation*math.sqrt(2)+electrode_size*math.sqrt(2)))
				m = int(max(width, height)/(minimum_separation*math.sqrt(2)+electrode_size*math.sqrt(2)))
				# print("electrode_size")
				# print(electrode_size)
				overlapping_area = (n) * (m) * minimum_separation * minimum_separation
				overlapping_area_score = overlapping_area/(width*height)
				# print("overlapping_area")
				# print(overlapping_area/(width*height))
				electrode_area = n * m * electrode_size * electrode_size * 2
				electrode_area_score = electrode_area/(width*height)
				# print("electrode_area")
				# print(electrode_area/(width*height))
				board = generate_pattern(width, height, electrode_size, minimum_separation)
				count = 0
				imbalance_percentages = []
				simulation_records = []
				for filename, test_shape in test_shapes:
					test_shape_height = test_shape.shape[0]
					test_shape_width = test_shape.shape[1]
					for rotation in simulation_shape_rotations:
						rotated_shape = ndimage.rotate(test_shape, rotation, reshape=True)
						for uniform_scale in range(1, simulation_max_uniform_sclae, shape_scale_precision):
							for x_scale in range(1, simulation_max_x_sclae,shape_scale_precision):
								if test_shape_width*uniform_scale*x_scale < width and test_shape_height*uniform_scale < height:
									# resized_shape = cv2.resize(rotated_shape, (test_shape_width*uniform_scale*x_scale, test_shape_height*uniform_scale), interpolation = cv2.INTER_AREA)
									# total = sum(sum(resized_shape))
									# # print(total)
									# # conv_matrix = conv2d(board, resized_shape, (shape_position_precision, shape_position_precision))
									# conv_matrix = conv2d(torch.from_numpy(board[None, None, : , :]), torch.from_numpy(resized_shape[None, None, : , :]), stride= (shape_position_precision, shape_position_precision))
									# conv_matrix = conv_matrix.cpu().detach().numpy()[0, 0, :, :]
									# # print(conv_matrix.shape)
									# imbalance_percentage = np.abs(conv_matrix/total)
									# # print(sum(sum(imbalance_percentage)))
									# average_imbalance_percentage = sum(sum(imbalance_percentage))/(conv_matrix.shape[0] * conv_matrix.shape[1])
									# simulation_parameter = (height, width, filename, rotation, uniform_scale, x_scale, average_imbalance_percentage)
									# imbalance_percentages.append(average_imbalance_percentage)
									# simulation_records.append(simulation_parameter)
									# count += (height - test_shape_height) * (width - test_shape_width)
									# print(simulation_parameter)
									count+=1
						print(count)
						count = 0
					break
				continue
				imbalance_percentages =  np.array(imbalance_percentages)
				cutting_score = np.mean(imbalance_percentages)
				# print(count)
					# for position in range()
				# result
				# cutting_score = 0
				result[electrode_size]["overall_score"].append(electrode_area_score - overlapping_area_score - cutting_score)
				result[electrode_size]["scores"].append((overlapping_area_score, electrode_area_score, cutting_score))
				result[electrode_size]["simulation_records"] = simulation_records
		result[electrode_size]["overall_score"] = np.array(result[electrode_size]["overall_score"])
		result[electrode_size]["overall_score"] = np.mean(result[electrode_size]["overall_score"])
		print("electrode_size")
		print(electrode_size)
		print("overall_score")
		print(overall_score)

	with open("simulation_records.json", "w") as outfile:
		json.dump(dictionary, outfile)

	
	result.sort(key=lambda y: y["overall_score"])
	print(result[-1])
	

if __name__ == '__main__':
	# test_shapes = load_shapes()
	# board = generate_pattern(1000, 1000, 500, minimum_separation)
	# height = board.shape[0]
	# width = board.shape[1]
	# for _, test_shape in test_shapes:
	# 	test_shape_height = test_shape.shape[0]
	# 	test_shape_width = test_shape.shape[1]
	# 	for rotation in simulation_shape_rotations:
	# 		rotated_shape = ndimage.rotate(test_shape, rotation, reshape=True)
	# 		for y_scale in range(2, int(1000/test_shape_height)):
	# 			for x_scale in range(2, int(1000/test_shape_width)):
	# 				resized_shape = cv2.resize(rotated_shape, (y_scale*test_shape_height, x_scale*test_shape_width), interpolation = cv2.INTER_AREA)
	# 				total = sum(sum(resized_shape))
	# 				# print(total)
	# 				# print(resized_shape.shape)
	# 				# conv_matrix = conv2d(board, resized_shape, (shape_position_precision, shape_position_precision))
	# 				# conv_matrix = signal.convolve2d(board, resized_shape, mode='valid')
	# 				conv_matrix = conv2d(torch.from_numpy(board[None, None, : , :]), torch.from_numpy(resized_shape[None, None, : , :]), stride= (shape_position_precision, shape_position_precision))
	# 				conv_matrix = conv_matrix.cpu().detach().numpy()[0, 0, :, :]
	# 				# conv_matrix = cv2.filter2D(board, -1, resized_shape)
	# 				# print(conv_matrix.shape)
	# 				np.savetxt("test.csv", conv_matrix, delimiter = ',', fmt='%1u')
	# 				imbalance_percentage = np.abs(conv_matrix/total)
	# 				average_imbalance_percentage = sum(sum(imbalance_percentage))/((height - test_shape_height) * (width - test_shape_width))
	# 				print(average_imbalance_percentage)
					
	# 				break
	# 			break
	# 		break
	# 	break
	main() 
	# generate_pattern(50, 50, 0, 0, 10, 1)
