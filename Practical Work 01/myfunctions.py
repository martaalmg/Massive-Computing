import multiprocessing as mp
from multiprocessing import Semaphore, Lock, Process, Pool
import numpy as np
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import ctypes
import myfunctions as my
from multiprocessing.sharedctypes import Value, Array, RawArray


def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(), dtype = np.uint8)

#Method which actually parallelize the processes
def parallel_filter(image, shared_image, filter_mask, numprocessors, lock):
    
    #image is the original image
    #shared_image is the shared memory where you will have to save the filtered image
    
    rows = image.shape[0] # Initialize how many rows we have
    v = range(rows) 
    
    with mp.Pool(processes = numprocessors, initializer = init_globalimage1,
                 initargs = [image, shared_image, filter_mask, lock]) as p:
        p.map(filter1, v) # Run the vector v in the function filter1        
    return
  
    
# Method to initialize the global variables which are going to be :
# - Image (original image to filter)
# - Shared_array which is where the filtered image is going to be stored
# - Filter_mask vector that has the filter
# - lock is the lock instance

def init_globalimage1(img, shared_array, filter_mask, lock):
    global image
    global shared_matrix    
    global my_filter
    global shared_space
    global global_lock
    
    image = img  
    my_filter = filter_mask
    global_lock = lock
    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(img.shape) 
    # Convert to an array and reshaped to a matrix^^^

    
# Check the dimensions of the filter given    
def dim_filter(array):  # array is the vector given by the function .shape

    if len(array) == 1: # Case when has one row and x columns. EX: (3,)
        rows = 1
        columns = array[0]
    elif len(array) == 2: # Case when has more than one row EX: (3,5)
        rows = array[0]
        columns = array[1]   
    
    return rows, columns #Gives the number of columns and rows has the filter_mask


# Method that filters the row given
def filter1(row): 
    # The input is the image row to filter (r)
    
    # We call the global variables we are going to use:
    global image
    global filtered_image    
    global my_filter
    global global_lock

    with shared_space.get_lock():
        (rows, cols, depth) = image.shape # from the global variable, gets the image shape   
        
    srow = image[row,:,:] # fetch the r row from the original image    
        
    # Declare the variables previous row (prow) and the next row (nrow)    
    # In this function we are also going to get the previous row to prow
    # and next row to nrow for the case of 5x5 or 5x1.

    if row == 0: # First row
            
        prevrow = image[row, :, :]
        prow = image[row, :, :]
        srow = image[row, :, :]
        nrow = image[row+1, :, :]
        nextrow = image[row+2, :, :]
        
    elif row == 1: # Second row
        prevrow = image[row-1, :, :]
        prow = image[row-1, :, :]
        srow = image[row, :, :]
        nrow = image[row+1, :, :]
        nextrow = image[row+2, :, :]
        
    elif row == rows-2: # Antepenultime row
        prevrow = image[row-2, :, :]
        prow = image[row-1, :, :]
        srow = image[row, :, :]
        nrow = image[row+1, :, :]
        nextrow = image[row+1, :, :]
        
    elif row == rows-1: # Last row
        prevrow = image[row-2, :, :]
        prow = image[row-1, :, :]
        srow = image[row, :, :]
        nrow = image[row, :, :]
        nextrow = image[row, :, :]
        
    else: # Middle row
        prevrow = image[row-2, :, :]
        prow = image[row-1, :, :]
        srow = image[row, :, :]
        nrow = image[row+1, :, :]
        nextrow = image[row+2, :, :]                   
       
    frow = np.zeros((cols, depth)) # Future filtered row

    shape_filter = my_filter.shape # Get the array to insert in the function dim_filter
    filter_row , filter_col = dim_filter(shape_filter) # Get the number of rows and columns of the filter
    
    # Having into account the dimensions of the filters we are going to have 9 cases:
    # In each case we run through every column and depth to be sure to filter each one of them 
    # and in each "cell" we compute the neighborhood matrix and we multiply it by the the filter.
    # As the neighborhood matrix has to be with the same dimensions of the filter_mask depending
    # on the case we have one matrix or another.

    
######################(5, 5) Case 1 #####################
    
    if filter_row == 5 and filter_col == 5:  
        
        for d in range(depth):
            for c in range(cols):
                if c == 0: #Caso que la column sea la primera 
                    n = np.array([ [ prevrow[c,d], prevrow[c,d], prevrow[c,d], prevrow[c+1,d], prevrow[c+2,d] ] 
                                  ,[ prow[c,d], prow[c,d], prow[c,d], prow[c+1,d], prow[c+2,d] ]
                                  ,[ srow[c,d], srow[c,d], srow[c,d], srow[c+1,d], srow[c+2,d] ]
                                  ,[ nrow[c,d], nrow[c,d], nrow[c,d], nrow[c+1,d], nrow[c+2,d] ]
                                  ,[ nextrow[c,d], nextrow[c,d], nextrow[c,d], nextrow[c+1,d], nextrow[c+2,d] ]])
                    
                elif c == 1: #Caso que la column sea la segunda 
                    n = np.array([ [ prevrow[c-1,d], prevrow[c-1,d], prevrow[c,d], prevrow[c+1,d], prevrow[c+2,d] ] 
                                  ,[ prow[c-1,d], prow[c-1,d], prow[c,d], prow[c+1,d], prow[c+2,d] ]
                                  ,[ srow[c-1,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+2,d] ]
                                  ,[ nrow[c-1,d], nrow[c-1,d], nrow[c,d], nrow[c+1,d], nrow[c+2,d] ]
                                  ,[ nextrow[c-1,d], nextrow[c-1,d], nextrow[c,d], nextrow[c+1,d], nextrow[c+2,d] ]])
                
                elif c > 1 and c < cols-2: #Caso que la column este por el medio
                    n = np.array([ [ prevrow[c-2,d], prevrow[c-1,d], prevrow[c,d], prevrow[c+1,d], prevrow[c+2,d] ] 
                                  ,[ prow[c-2,d], prow[c-1,d], prow[c,d], prow[c+1,d], prow[c+2,d] ]
                                  ,[ srow[c-2,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+2,d] ]
                                  ,[ nrow[c-2,d], nrow[c-1,d], nrow[c,d], nrow[c+1,d], nrow[c+2,d] ]
                                  ,[ nextrow[c-2,d], nextrow[c-1,d], nextrow[c,d], nextrow[c+1,d], nextrow[c+2,d] ]])
                
                elif c == cols-1: #Caso que la column sea la ultima 
                    n = np.array([ [ prevrow[c-2,d], prevrow[c-1,d], prevrow[c,d], prevrow[c,d], prevrow[c,d] ] 
                                  ,[ prow[c-2,d], prow[c-1,d], prow[c,d], prow[c,d], prow[c,d] ]
                                  ,[ srow[c-2,d], srow[c-1,d], srow[c,d], srow[c,d], srow[c,d] ]
                                  ,[ nrow[c-2,d], nrow[c-1,d], nrow[c,d], nrow[c,d], nrow[c,d] ]
                                  ,[ nextrow[c-2,d], nextrow[c-1,d], nextrow[c,d], nextrow[c,d], nextrow[c,d] ]])
                    
                elif c == cols-2: #Caso que la column sea la penultima 
                    n = np.array([ [ prevrow[c-2,d], prevrow[c-1,d], prevrow[c,d], prevrow[c+1,d], prevrow[c+1,d] ] 
                                  ,[ prow[c-2,d], prow[c-1,d], prow[c,d], prow[c+1,d], prow[c+1,d] ]
                                  ,[ srow[c-2,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+1,d] ]
                                  ,[ nrow[c-2,d], nrow[c-1,d], nrow[c,d], nrow[c+1,d], nrow[c+1,d] ]
                                  ,[ nextrow[c-2,d], nextrow[c-1,d], nextrow[c,d], nextrow[c+1,d], nextrow[c+1,d] ]])
                    
                value = np.sum(my_filter * n)
                frow[c][d] = value

######################(5, 3) Case 2 #####################    
        
    elif filter_row == 5 and filter_col == 3: 
        for d in range(depth):
            for c in range(cols):
                if c == 0: #Caso que la columna sea la primera 
                    n = np.array([ [ prevrow[c,d], prevrow[c,d], prevrow[c+1,d] ] 
                                  ,[ prow[c,d], prow[c,d], prow[c+1,d] ]
                                  ,[ srow[c,d], srow[c,d], srow[c+1,d] ]
                                  ,[ nrow[c,d], nrow[c,d], nrow[c+1,d] ]
                                  ,[ nextrow[c,d], nextrow[c,d], nextrow[c+1,d] ]])
                                  
                elif c > 0 and c < cols-1: #Caso que la columna este por el medio
                    n = np.array([ [ prevrow[c-1,d], prevrow[c,d], prevrow[c+1,d] ] 
                                  ,[ prow[c-1,d], prow[c,d], prow[c+1,d] ]
                                  ,[ srow[c-1,d], srow[c,d], srow[c+1,d] ]
                                  ,[ nrow[c-1,d], nrow[c,d], nrow[c+1,d] ]
                                  ,[ nextrow[c-1,d], nextrow[c,d], nextrow[c+1,d] ]])
                    
                elif c == cols-1: #Caso que la columna sea la ultima
                    n = np.array([ [ prevrow[c-1,d], prevrow[c,d], prevrow[c,d] ] 
                                  ,[ prow[c-1,d], prow[c,d], prow[c,d] ]
                                  ,[ srow[c-1,d], srow[c,d], srow[c,d] ]
                                  ,[ nrow[c-1,d], nrow[c,d], nrow[c,d] ]
                                  ,[ nextrow[c-1,d], nextrow[c,d], nextrow[c,d] ]])
                    
                value = np.sum(my_filter * n)
                frow[c][d] = value
                
######################(5, 1) Case 3 #####################                               
           
    elif filter_row == 5 and filter_col == 1: # (5, 1) Case 3
        for d in range(depth):
            for c in range(cols):
                n = np.array([ [prevrow[c,d]],[prow[c,d]] ,[srow[c,d]] ,[nrow[c,d]],[nextrow[c,d] ]])   
                
                value = np.sum(my_filter * n)
                frow[c][d] = value

######################(3, 5) Case 4 #####################                
                
    elif filter_row == 3 and filter_col == 5:
        for d in range(depth):
            for c in range(cols):
                if c == 0: #Caso que la column sea la primera 
                    n = np.array([ [ prow[c,d], prow[c,d], prow[c,d], prow[c+1,d], prow[c+2,d] ]
                                  ,[ srow[c,d], srow[c,d], srow[c,d], srow[c+1,d], srow[c+2,d] ]
                                  ,[ nrow[c,d], nrow[c,d], nrow[c,d], nrow[c+1,d], nrow[c+2,d] ]])
                    
                elif c == 1: #Caso que la column sea la segunda 
                    n = np.array([ [ prow[c-1,d], prow[c-1,d], prow[c,d], prow[c+1,d], prow[c+2,d] ]
                                  ,[ srow[c-1,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+2,d] ]
                                  ,[ nrow[c-1,d], nrow[c-1,d], nrow[c,d], nrow[c+1,d], nrow[c+2,d] ]])                                        
                elif c > 1 and c < cols-2: #Caso que la column este por el medio
                    n = np.array([ [ prow[c-2,d], prow[c-1,d], prow[c,d], prow[c+1,d], prow[c+2,d] ]
                                  ,[ srow[c-2,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+2,d] ]
                                  ,[ nrow[c-2,d], nrow[c-1,d], nrow[c,d], nrow[c+1,d], nrow[c+2,d] ]])                                       
                elif c == cols-1: #Caso que la column sea la ultima 
                    n = np.array([ [ prow[c-2,d], prow[c-1,d], prow[c,d], prow[c,d], prow[c,d] ]
                                  ,[ srow[c-2,d], srow[c-1,d], srow[c,d], srow[c,d], srow[c,d] ]
                                  ,[ nrow[c-2,d], nrow[c-1,d], nrow[c,d], nrow[c,d], nrow[c,d] ]])
                                                           
                elif c == cols-2: #Caso que la column sea la penultima 
                    n = np.array([ [ prow[c-2,d], prow[c-1,d], prow[c,d], prow[c+1,d], prow[c+1,d] ]
                                  ,[ srow[c-2,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+1,d] ]
                                  ,[ nrow[c-2,d], nrow[c-1,d], nrow[c,d], nrow[c+1,d], nrow[c+1,d] ]])                                        
            value = np.sum(my_filter * n)
            frow[c][d] = value
                
######################(1, 5) Case 5 ##################### 

    elif filter_row == 1 and filter_col == 5: 
        for d in range(depth):
            for c in range(cols):
                if c == 0: #Caso que la column sea la primera
                    n = np.array([ [srow[c,d], srow[c,d], srow[c,d], srow[c+1,d], srow[c+2,d]] ])
                    
                elif c == 1: #Caso que la column sea la segunda 
                    n = np.array([ [srow[c-1,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+2,d]] ])                                        
                elif c > 1 and c < cols-2: #Caso que la column este por el medio
                    n = np.array([ [srow[c-2,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+2,d]] ])                                       
                elif c == cols-1: #Caso que la column sea la ultima 
                    n = np.array([[srow[c-2,d], srow[c-1,d], srow[c,d], srow[c,d], srow[c,d]] ])
                                                           
                elif c == cols-2: #Caso que la column sea la penultima 
                    n = np.array([[srow[c-2,d], srow[c-1,d], srow[c,d], srow[c+1,d], srow[c+1,d]] ])                                        
            value = np.sum(my_filter * n)
            frow[c][d] = value
        
        
######################(3, 3) Case 6 #####################         
            
    elif filter_row == 3 and filter_col == 3: 
        for d in range(depth):
            for c in range(cols):
                if c == 0: #Caso que la column sea la primera
                    n = np.array([ [prow[c,d], prow[c,d], prow[c+1,d] ]
                                  ,[srow[c,d], srow[c,d], srow[c+1,d] ]
                                  ,[nrow[c,d], nrow[c,d], nrow[c+1,d] ]])
                    
                elif c > 0 and c < cols-1: #Caso que la column este por el medio
                    n = np.array([ [prow[c-1,d], prow[c,d], prow[c+1,d] ]
                                  ,[srow[c-1,d], srow[c,d], srow[c+1,d] ]
                                  ,[nrow[c-1,d], nrow[c,d], nrow[c+1,d] ]])
                    
                elif c == cols-1: #Caso que la column sea la ultima 
                    n = np.array([ [prow[c-1,d], prow[c,d], prow[c,d] ]
                                  ,[srow[c-1,d], srow[c,d], srow[c,d] ]
                                  ,[nrow[c-1,d], nrow[c,d], nrow[c,d] ]])
                    
                value = np.sum(my_filter * n)
                frow[c][d] = value                       
                
######################(3, 1) Case 7 #####################                 
        
    elif filter_row == 3 and filter_col == 1:
        for d in range(depth):
            for c in range(cols):
                n = np.array([ [prow[c,d]],[srow[c,d]],[nrow[c,d]] ])
                
                value = np.sum(my_filter * n)
                frow[c][d] = value
                 
                                                                
######################(1, 3) Case 8 #####################                 
        
    elif filter_row == 1 and filter_col == 3: 
        for d in range(depth):
            for c in range(cols):
                if c == 0: #Caso que la column sea la primera
                    n = np.array([ [srow[c,d], srow[c,d], srow[c+1,d]] ])
                    
                elif c > 0 and c < cols-1: #Caso que la column este por el medio
                    n = np.array([ [srow[c-1,d], srow[c,d], srow[c+1,d]] ])
                    
                elif c == cols-1: #Caso que la column sea la ultima 
                    n = np.array([ [srow[c-1,d], srow[c,d], srow[c,d]] ]) 
                   
                value = np.sum(my_filter * n)
                frow[c][d] = value
                
        
######################(1, 1) Case 9 #####################   

    elif filter_row == 1 and filter_col == 1:  
        for d in range(depth):
            for c in range(cols):
                n = np.array([ [srow[c,d]] ])
                value = my_filter * n
                frow[c][d] = value
                
    shared_matrix[row, :, :] = frow # We store the filtered row in the shared memory

 
 
    

 
            