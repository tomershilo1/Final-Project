import numpy as np
from spectral import *
import matplotlib.pyplot as plt
import time

start_time = time.time()

def read_hyperData(fileName):
    img = open_image(fileName)  
    return img.load()

def generate_3Ddata():
    xDim = 5
    yDim = 5
    zDim = 6
    random_values = np.random.rand(zDim, xDim, yDim)
    return random_values

def local_mean_zxy(data):
# input: 3D data
# output: 3D mean
    zDim, xDim, yDim = data.shape
    local_means = np.zeros(( zDim, xDim, yDim))
    for z in range(zDim):
        for x in range(xDim):
            for y in range(yDim):
                #need to update mean calculation to 1 neighbor
                neighbors = []
                if x == 0:
                    neighbors.append(data[z, x+1, y])
                elif x == xDim - 1:
                    neighbors.append(data[z, x-1, y])
                else:
                    neighbors.append(data[z, x-1, y])
                    neighbors.append(data[z, x+1, y])
                    
                if y == 0:
                    neighbors.append(data[z, x, y+1])
                elif y == yDim - 1:
                    neighbors.append(data[z, x, y-1])
                else:
                    neighbors.append(data[z, x, y-1])
                    neighbors.append(data[z, x, y+1])
                # Calculate the mean of the neighboring coordinates
                if neighbors:
                    local_means[z, x, y] = np.mean(neighbors)
    return local_means

def local_mean_xyz(data,selected_band):
# input: 3D data
# output: 3D mean
    xDim, yDim, zDim = data.shape
    local_means = np.zeros((xDim, yDim, selected_band+1))
    for z in range(selected_band+1):
        for x in range(xDim):
            for y in range(yDim):
                # calculating according to all previous data
                neighbors = []
                if x == 0:
                    if y == 0:
                        local_means[x, y, z] = data[x, y, z]
                    else:
                        neighbors.append(data[x, y-1, z])
                elif y == yDim - 1:
                    # if x == 0:  is already taken care of
                    neighbors.append(data[x - 1, y, z])
                    neighbors.append(data[x, y - 1, z])
                    neighbors.append(data[x - 1, y - 1, z])
                elif y == 0:
                    # if x == 0:  is already taken care of
                    neighbors.append(data[x - 1, y, z])
                    neighbors.append(data[x - 1, y+1, z])
                else:
                    neighbors.append(data[x - 1, y, z])
                    neighbors.append(data[x - 1, y - 1, z])
                    neighbors.append(data[x, y - 1, z])
                    neighbors.append(data[x - 1, y + 1, z])
                # Calculate the mean of the neighboring coordinates
                if neighbors:
                    local_means[x, y, z] = np.mean(neighbors)
    return local_means

def difference_matrix_zxy(data,mean):
# input: 3D data and 3D mean
# output: 4D difference matrix
    zDim, xDim, yDim = data.shape
    difference_matrix = np.zeros((zDim, xDim, yDim))
    for z in range(zDim):
        for x in range(xDim):
            for y in range(yDim):
                if x == 0 or y == 0:
                    continue  # Neglecting the first iteration per first row X and and first column Y
                elif z < 2 :
                    continue  # Neglecting the first 2 iterations Z per first bands
                else:
                    differnce_vector = [#directional differneces
                                    data[z, x,y-1]-mean[z, x,y],
                                    data[z, x-1,y]-mean[z, x,y],
                                    data[z, x-1,y-1]-mean[z, x,y],
                                    #central local differences 
                                    data[z-1,x,y]-mean[z-1,x,y],
                                    data[z-2,x,y]-mean[z-2,x,y]]
                    #zero index is selected arbitrarly due to dimention problem
                    #need to implement the weight vector here
                    difference_matrix[z, x, y] = differnce_vector[0]
    return difference_matrix

def difference_matrix_xyz(data,mean, selected_band, weight_vector):
# input: 3D data and 3D mean
# output: 4D difference matrix
    xDim, yDim, zDim = data.shape
    difference_matrix = np.zeros((xDim, yDim, selected_band+1))
    for z in range(selected_band+1):
        for x in range(xDim):
            for y in range(yDim):
                difference_vector = [0,0,0,0,0]
                if x == 0:
                    #directional differneces
                    difference_vector[0] = data[x,y-1,z]-mean[x,y,z]
                elif y == 0:
                    #directional differneces
                    difference_vector[1] = data[x-1,y,z]-mean[x,y,z]
                else:
                    #directional differneces
                    difference_vector[0] = data[x,y-1,z]-mean[x,y,z]
                    difference_vector[1] = data[x-1,y,z]-mean[x,y,z]
                    difference_vector[2] = data[x-1,y-1,z]-mean[x,y,z]
                if z == 0:
                    continue
                elif z == 1 :
                    #central differences
                    difference_vector[3] = data[x,y,z-1]-mean[x,y,z]
                else:
                    difference_vector[3] = data[x,y,z-1]-mean[x,y,z]
                    difference_vector[4] = data[x,y,z-2]-mean[x,y,z]
                #zero index is selected arbitrarly due to dimention problem
                #need to implement the weight vector here, for now take [3] arbitrarly
                difference_matrix[x, y, z] = np.dot(weight_vector,difference_vector)
    return difference_matrix

def difference_matrix_frame_deltas_byX_1(data,mean, selected_band, weight_vector):
    # input: 3D data and 3D mean
    # output: 4D difference matrix
    xDim, yDim, zDim = data.shape
    difference_matrix = np.zeros((xDim, yDim, selected_band+1))
    for z in range(selected_band+1):
        for x in range(xDim):
            for y in range(yDim):
                difference_vector = [0,0,0,0,0]
                dif_set = False
                if x == 0:
                    if y == 0:
                        continue
                    else:
                    #first row deltas
                        difference_matrix[x, y, z] = data[x,y,z] - data[x,y-1,z]
                        dif_set = True
                elif y == 0:
                    #first column deltas
                        #difference_matrix[x, y, z] = np.mean([data[x-1,y,z],data[x-1,y+1,z]]) - data[x,y,z]
                        difference_matrix[x, y, z] = data[x,y,z] - data[x-1,y,z]
                        dif_set = True
                else:
                    #directional differneces
                    difference_vector[0] = data[x,y-1,z]-mean[x,y,z]
                    difference_vector[1] = data[x-1,y,z]-mean[x,y,z]
                    difference_vector[2] = data[x-1,y-1,z]-mean[x,y,z]
                if z == 1 :
                    #central differences
                    difference_vector[3] = data[x,y,z-1]-mean[x,y,z]
                if z > 1:
                    difference_vector[3] = data[x,y,z-1]-mean[x,y,z]
                    difference_vector[4] = data[x,y,z-2]-mean[x,y,z]
                #zero index is selected arbitrarly due to dimention problem
                #need to implement the weight vector here, for now take [3] arbitrarly
                if dif_set == False:

                    # need to implement here minimization functionality with respect to Sxyz

                    difference_matrix[x, y, z] = np.dot(weight_vector,difference_vector)
    return difference_matrix

def difference_matrix_by_Sxyz(data,mean, selected_band, weight_vector):
    # input: 3D data and 3D mean
    # output: 4D difference matrix
    xDim, yDim, zDim = data.shape
    difference_matrix = np.zeros((xDim, yDim, selected_band+1))
    for z in range(selected_band+1):
        for x in range(xDim):
            for y in range(yDim):
                difference_vector = [0,0,0,0,0]
                dif_set = False
                if x == 0:
                    if y == 0:
                        continue
                    else:
                    #first row deltas
                        difference_matrix[x, y, z] = data[x,y,z] - data[x,y-1,z]
                        dif_set = True
                elif y == 0:
                    #first column deltas
                        #difference_matrix[x, y, z] = np.mean([data[x-1,y,z],data[x-1,y+1,z]]) - data[x,y,z]
                        difference_matrix[x, y, z] = data[x,y,z] - data[x-1,y,z]
                        dif_set = True
                else:
                    difference_matrix[x, y, z] = data[x,y,z] - mean[x,y,z]
    return difference_matrix

def difference_matrix_by_2NN_including2first_bands(data,mean, selected_band, weight_vector,min_operator):
        # input: 3D data and 3D mean
    # output: 4D difference matrix
    xDim, yDim, zDim = data.shape
    difference_matrix = np.zeros((xDim, yDim, selected_band+1))
    difference_matrix[:, :, 0] = data[:, :, 0].squeeze()
    difference_matrix[:, :, 1] = data[:, :, 1].squeeze()
    for z in range(2,selected_band+1):
            for x in range(xDim):
                for y in range(yDim):
                    difference_vector = [0,0,0,0,0]
                    dif_set = False
                    if x == 0:
                        if y == 0:
                            continue
                        else:
                        #first row deltas
                            difference_matrix[x, y, z] = data[x,y,z] - data[x,y-1,z]
                            dif_set = True
                    elif y == 0:
                        #first column deltas
                            #difference_matrix[x, y, z] = np.mean([data[x-1,y,z],data[x-1,y+1,z]]) - data[x,y,z]
                            difference_matrix[x, y, z] = data[x,y,z] - data[x-1,y,z]
                            dif_set = True
                    else:
                        #directional differneces
                        difference_vector[0] = data[x,y-1,z]-mean[x,y,z]
                        difference_vector[1] = data[x-1,y,z]-mean[x,y,z]
                        difference_vector[2] = data[x-1,y-1,z]-mean[x,y,z]
                    if z == 1 :
                        #central differences
                        difference_vector[3] = data[x,y,z-1]-mean[x,y,z]
                    if z > 1:
                        difference_vector[3] = data[x,y,z-1]-mean[x,y,z]
                        difference_vector[4] = data[x,y,z-2]-mean[x,y,z]
                    #zero index is selected arbitrarly due to dimention problem
                    #need to implement the weight vector here, for now take [3] arbitrarly
                    if dif_set == False:
                        if min_operator == 1: 
                        # need to implement here minimization functionality with respect to Sxyz
                            difference_matrix[x, y, z] = minimize_single_delta(data[x,y,z],difference_vector,mean[x,y,z])
                        # difference_matrix[x, y, z] = np.dot(weight_vector,difference_vector)
                        else:
                            difference_matrix[x, y, z] = minimize_two_closest_delta(data[x,y,z],difference_vector,mean[x,y,z])
    return difference_matrix

def minimize_single_delta(Sxyz, difference_vector, mean):
    min = 999
    output_diff = difference_vector[0]
    for diff in difference_vector:
        # check how close we get to Sxyz
        if abs(Sxyz - (mean + diff)) < min:
            min = abs(Sxyz - (mean + diff))
            output_diff = diff
    return output_diff

def minimize_two_closest_delta(Sxyz, difference_vector, mean):
    min1 = 999
    min2 = 999
    diff1 = 0
    diff2 = 0
    output_diff = difference_vector[0]
    for diff in difference_vector:
        # check how close we get to Sxyz
        if abs(Sxyz - (mean + diff)) < min1:
            if abs(Sxyz - (mean + diff)) < min2:
                min2 = abs(Sxyz - (mean + diff))
                diff2 = diff
            else:
                min1 = abs(Sxyz - (mean + diff))
                diff1 = diff
        output_diff = 0.2*diff1+0.8*diff2
    return output_diff

def construct_recievedImage_byZ_1(difference_matrix, initial_data, weights_vector, mean):
    # output: image at the reciever
    # in the case of initial frame is given and weight is previous band
    xDim, yDim, zDim = difference_matrix.shape
    recieved_image = np.zeros((xDim, yDim, zDim))
    # mean = np.zeros((xDim, yDim, zDim))
    recieved_image[:,:,0] = initial_data.squeeze()
    for z in range(1,zDim):
        previous_band = recieved_image[:,:,z-1]
        for x in range(xDim):
            for y in range(yDim):
                # construct image by previos band
                if z>1:
                    neighbors = [mean[x,y,z], mean[x,y,z], mean[x,y,z], mean[x,y,z-1], mean[x,y,z-2] ]
                else:
                # z==1
                    neighbors = [mean[x,y,z], mean[x,y,z], mean[x,y,z], mean[x,y,z-1], mean[x,y,z] ]
                #else:
                # z==0
                #    print("does it happen ?")
                #    neighbors = [mean[x,y,z], mean[x,y,z], mean[x,y,z], mean[x,y,z], mean[x,y,z] ]
                #recieved_image[x,y,z] = weights_reconstruct(weights_vector, difference_matrix[x,y,z],neighbors)
                recieved_image[x,y,z] = previous_band[x,y] - difference_matrix[x,y,z] 
    return recieved_image

def construct_recievedImage_byX_1(difference_matrix, initial_data, weights_vector):
    # output: image at the reciever
    # in the case of initial pixel is given and weight is previous pixel x axis
    xDim, yDim, zDim = difference_matrix.shape
    recieved_image = np.zeros((xDim, yDim, zDim))
    mean = 0
    # mean = np.zeros((xDim, yDim, zDim))
    # recieved_image[:,:,0] = construct_top_row(difference_matrix[])
    initial_pixel = initial_data[0]
    for z in range(zDim):
        #recieved_image[x,y,z] = construct_top_row(difference_matrix[:,:,z-1],initial_pixel)
        # need to redefine initial data case for the next iteration
        for x in range(xDim):
            for y in range(yDim):
                neighbors = []
                if x == 0:
                    if y == 0:
                        neighbors.append(initial_pixel)
                        #recieved_image[x,y,z] = initial_pixel
                        #print(recieved_image[x,y,z])
                    else:
                        neighbors.append(recieved_image[x, y-1, z])
                elif y == yDim - 1:
                    # if x == 0:  is already taken care of
                    neighbors.append(recieved_image[x - 1, y, z])
                    neighbors.append(recieved_image[x, y - 1, z])
                    neighbors.append(recieved_image[x - 1, y - 1, z])
                elif y == 0:
                    # if x == 0:  is already taken care of
                    neighbors.append(recieved_image[x - 1, y, z])
                    # neighbors.append(recieved_image[x - 1, y+1, z])
                else:
                    neighbors.append(recieved_image[x - 1, y, z])
                    neighbors.append(recieved_image[x - 1, y - 1, z])
                    neighbors.append(recieved_image[x, y - 1, z])
                    neighbors.append(recieved_image[x - 1, y + 1, z])
                mean = np.mean(neighbors)
                recieved_image[x,y,z] = mean + difference_matrix[x,y,z]
                # recieved_image[x,y,z] = data[x,y-1,z] + data[x,y,z] - data[x,y-1,z]
                
                if z < zDim: 
                    initial_pixel = initial_data[z+1] 
    return recieved_image

def construct_recievedImage_byS(difference_matrix, initial_data):
    # output: image at the reciever
    # in the case of initial pixel is given and weight is previous pixel x axis
    xDim, yDim, zDim = difference_matrix.shape
    recieved_image = np.zeros((xDim, yDim, zDim))
    mean = 0
    # mean = np.zeros((xDim, yDim, zDim))
    # recieved_image[:,:,0] = construct_top_row(difference_matrix[])
    initial_pixel = initial_data[0]
    for z in range(zDim):
        #recieved_image[x,y,z] = construct_top_row(difference_matrix[:,:,z-1],initial_pixel)
        # need to redefine initial data case for the next iteration
        for x in range(xDim):
            for y in range(yDim):
                neighbors = []
                if x == 0:
                    if y == 0:
                        neighbors.append(initial_pixel)
                        #recieved_image[x,y,z] = initial_pixel
                        #print(recieved_image[x,y,z])
                    else:
                        neighbors.append(recieved_image[x, y-1, z])
                elif y == yDim - 1:
                    # if x == 0:  is already taken care of
                    neighbors.append(recieved_image[x - 1, y, z])
                    neighbors.append(recieved_image[x, y - 1, z])
                    neighbors.append(recieved_image[x - 1, y - 1, z])
                elif y == 0:
                    # if x == 0:  is already taken care of
                    neighbors.append(recieved_image[x - 1, y, z])
                    # neighbors.append(recieved_image[x - 1, y+1, z])
                else:
                    neighbors.append(recieved_image[x - 1, y, z])
                    neighbors.append(recieved_image[x - 1, y - 1, z])
                    neighbors.append(recieved_image[x, y - 1, z])
                    neighbors.append(recieved_image[x - 1, y + 1, z])
                mean = np.mean(neighbors)
                recieved_image[x,y,z] = mean + difference_matrix[x,y,z]
                # recieved_image[x,y,z] = mean + data[x,y,z] - mean[x,y,z]
                if z < zDim: 
                    initial_pixel = initial_data[z+1] 
    return recieved_image

def construct_recievedImage_byZ_2(difference_matrix, initial_data, weights_vector):
    # output: image at the reciever
    # in the case of initial frame is given and weight is previous band
    xDim, yDim, zDim = difference_matrix.shape
    recieved_image = np.zeros((xDim, yDim, zDim))
    # mean = np.zeros((xDim, yDim, zDim))
    recieved_image[:,:,0] = difference_matrix[:,:,0]
    recieved_image[:,:,1] = difference_matrix[:,:,1]
    for z in range(2,zDim):
        initial_pixel = initial_data[z]
        for x in range(xDim):
            for y in range(yDim):
                neighbors = []
                if x == 0:
                    if y == 0:
                        neighbors.append(initial_pixel)
                        #recieved_image[x,y,z] = initial_pixel
                        #print(recieved_image[x,y,z])
                    else:
                        neighbors.append(recieved_image[x, y-1, z])
                elif y == yDim - 1:
                    # if x == 0:  is already taken care of
                    neighbors.append(recieved_image[x - 1, y, z])
                    neighbors.append(recieved_image[x, y - 1, z])
                    neighbors.append(recieved_image[x - 1, y - 1, z])
                elif y == 0:
                    # if x == 0:  is already taken care of
                    neighbors.append(recieved_image[x - 1, y, z])
                    # neighbors.append(recieved_image[x - 1, y+1, z])
                else:
                    neighbors.append(recieved_image[x - 1, y, z])
                    neighbors.append(recieved_image[x - 1, y - 1, z])
                    neighbors.append(recieved_image[x, y - 1, z])
                    neighbors.append(recieved_image[x - 1, y + 1, z])
                mean = np.mean(neighbors)
                recieved_image[x,y,z] = mean + difference_matrix[x,y,z] 
    return recieved_image

def construct_top_row(difference_frame, initial_data):
    xDim, yDim = difference_frame.shape
    mean = np.zeros((xDim, yDim))
    for y in range(yDim):
        if y == 0:
            # initial data is top left-corner pixel each band
            mean[0, y] = initial_data
        else:
            mean[0, y] = mean[0, y-1] - difference_frame[0,y]
    return mean

#def weights_reconstruct(weights_vector, difference_vector, neighbors):
    # neighbors = [ mu_xyz, mu_xyz, mu_xyz, previous_pixl, previous_prevous_pixl ]
#    return np.dot(weights_vector , np.subtract(neighbors,difference_vector))

def weights_reconstruct(weights_vector, difference_vector, neighbors):
    # neighbors = [ mu_xyz, mu_xyz, mu_xyz, previous_pixl, previous_prevous_pixl ]
    return np.dot(weights_vector ,neighbors)-difference_vector

# Generate the 3D array
random_data =  generate_3Ddata()

# Read hyper filer
hyper_data = read_hyperData('92AV3C.lan')
selected_band = 30  # You can change this to select a different band
weights_vector = [1,0,0,0,0]

# Calculate local means
local_means_random = local_mean_zxy(random_data)
local_means_hyper = local_mean_xyz(hyper_data, selected_band)

#Calculate 4D difference matrix
# difference_variable_random = difference_matrix_zxy(random_data, local_means_random)
difference_variable_hyper1 = difference_matrix_by_Sxyz(hyper_data, local_means_hyper, selected_band, weights_vector)
difference_variable_hyper2 = difference_matrix_by_2NN_including2first_bands(hyper_data, local_means_hyper, selected_band, weights_vector,1)
difference_variable_hyper3 = difference_matrix_by_2NN_including2first_bands(hyper_data, local_means_hyper, selected_band, weights_vector,0)

# Print the first local mean for each zDim
#print("Local Means (First row excluded):")
''' this is the bits per pixel before integrated to the calculation for entire image
def bits_per_pixel(matrix):
    max_value = np.max(matrix)
    if max_value == 0:
        return 1
    return int(np.ceil(np.log2(max_value + 1)))
'''
def bits_per_pixel_image(image):
    # Flatten the image to a 1D array for easier iteration
    flattened_image = image.flatten()
    
    # Calculate bits for each pixel value
    bits_per_pixel = np.zeros_like(flattened_image, dtype=int)
    
    # Calculate bits for non-zero positive values
    non_zero_positive_mask = (flattened_image > 0)
    if np.any(non_zero_positive_mask):
        bits_per_pixel[non_zero_positive_mask] = np.ceil(np.log2(flattened_image[non_zero_positive_mask] + 1)).astype(int)
    
    # Handle zero and negative values separately
    bits_per_pixel[flattened_image <= 0] = 1
    
    # Sum all the bits needed for the entire image
    total_bits = np.sum(bits_per_pixel)
    
    return total_bits

def compression_ratio(original_image, residuals):
    original_bits = bits_per_pixel_image(original_image)
    residuals_bits = bits_per_pixel_image(residuals)
    return residuals_bits / original_bits 

# Calculate compression ratio
compression_ratio_value1 = compression_ratio(hyper_data[:, :, selected_band], difference_variable_hyper1[:, :, selected_band])
compression_ratio_value2 = compression_ratio(hyper_data[:, :, selected_band], difference_variable_hyper2[:, :, selected_band])
compression_ratio_value3 = compression_ratio(hyper_data[:, :, selected_band], difference_variable_hyper3[:, :, selected_band])

print(f"Compression Ratio: {compression_ratio_value1:.2f}")


selected_image = hyper_data[:, :, selected_band].squeeze()
#initial_frame = hyper_data[:, :, 0]
initial_pixels = hyper_data[0,0,:].squeeze()
top_row_filled = construct_top_row(difference_variable_hyper1[:,:,selected_band],selected_image[0,0])
recieved_image = construct_recievedImage_byS(difference_variable_hyper1,initial_pixels)
recieved_image_dif_vector1 = construct_recievedImage_byZ_2(difference_variable_hyper2,initial_pixels,weights_vector)
recieved_image_dif_vector2 = construct_recievedImage_byZ_2(difference_variable_hyper3,initial_pixels,weights_vector)



print(selected_image[0:5,0:5])
print(recieved_image[0:5,0:5,selected_band])
print(difference_variable_hyper1[0:5,0:5,selected_band])
print(local_means_hyper[0:5,0:5,selected_band])
# print(hyper_data[0:5,0:5,selected_band-1])


#print(top_row_filled[0:5,0:5])

## print constructed img vs hyper data img
figure, axis = plt.subplots(2,2)

axis[0,0].imshow(recieved_image[:,:,selected_band])  # Displaying the selected band as grayscale, adjust as needed
axis[0,0].set_title("Constructed Image \n By S Version 1")

## Display the image using imshow
axis[0,1].imshow(selected_image)  # Displaying the selected band as grayscale, adjust as needed
axis[0,1].set_title("Original Image")

axis[1,0].imshow(recieved_image_dif_vector1[:,:,selected_band])  # Displaying the selected band as grayscale, adjust as needed
axis[1,0].set_title("Constructed Image \n By Difference Vector version 1")

axis[1,1].imshow(recieved_image_dif_vector2[:,:,selected_band])  # Displaying the selected band as grayscale, adjust as needed
axis[1,1].set_title("Constructed Image \n By Difference Vector version 2")

axis[1,0].text(0.5, -0.3, "Compression ratio:%f"%(compression_ratio_value2), ha='center', va='top', transform=axis[1,0].transAxes, fontsize=12) # need to insert the compression value based on function that calculates it
axis[1,1].text(0.5, -0.3, "Compression ratio:%f"%(compression_ratio_value3), ha='center', va='top', transform=axis[1,1].transAxes, fontsize=12) # need to insert the compression value based on function that calculates it
axis[0,0].text(0.5, -0.3, "Compression ratio:%f"%(compression_ratio_value1), ha='center', va='top', transform=axis[0,0].transAxes, fontsize=12) # need to insert the compression value based on function that calculates it
plt.suptitle("Image Band %d out of %d"%(selected_band,hyper_data.shape[2]))
plt.tight_layout()
plt.show()


end_time = time.time()
print(f"Runtime: {end_time - start_time} seconds")

#print(random_data)
# print(local_means)
# print(difference_variable)
#for z in range(local_means.shape[2]):
#   print(f"ZDim {z+1}:")
#  print(local_means[:,:,z]