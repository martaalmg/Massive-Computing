#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
#
import numpy as np


# In[ ]:




from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes



# In[ ]:


#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

#this function creates the instance of Value data type and initialize it to 0
def dot_init(g_A):
    global A 
    A = g_A #We create a variable of type "double"           
    
    
def shared_dot_1(V):
    #This code is wrong!!!
    for f in V:
        A.value += f[0]*f[1]
    
def shared_dot_2(V):
    #This code is wrong!!!
    with A.get_lock():
        for f in V:
            A.value += f[0]*f[1]
    
def shared_dot_3(V):
    #This code is wrong!!!
    a=0
    for f in V:
        a += f[0]*f[1]
    with A.get_lock():
        A.value += a
    


#This function initialize the global shared memory data

def pool_init(shared_array_,srcimg, imgfilter):
    #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
    #srcimg: is the original image
    #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
    
    #We defines the local process memory reference for shared memory space
    global shared_space
    #Here we define the numpy matrix handler
    global shared_matrix
    
    #Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
    global image
    global my_filter
    
    #here, we initialize the global read only memory data
    image=srcimg
    my_filter=imgfilter
    size = image.shape
    
    #Assign the shared memory  to the local reference
    shared_space = shared_array_
    #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)


# In[ ]:


#this function just copy the original image to the global r/w shared  memory 
def parallel_shared_imagecopy(row):
    global image
    global my_filter
    global shared_space    
    # with this instruction we lock the shared memory space, avoidin other parallel processes tries to write on it
    with shared_space.get_lock():
        #while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[row,:,:]=image[row,:,:]
    return


# In[ ]:


def edge_filter(row): # The input is the image row to filter (r)
    
    global image # image is the global memory array
    global my_filter # my_filter is the filter shape to apply to the image  
    global shared_space 
    #shared space is where we are going to save the shared matrix which is in shared memory
       
    
    (rows,cols,depth) = image.shape # from the global variaable, gets the image shape
    
    srow=image[row,:,:] # fetch the r row from the original image
    
    # Now it starts to declare the variables previous row (prow) and the next row (nrow)
    
    if ( row>0 ): # In case the row is not the first (in the border)
        
        prow=image[row-1,:,:] # We just declare that the previous row could be perfectly the before one          
    else:
        
        prow=image[row,:,:] # If that is not the case and it is the case where is the first row then the previous row does not exist and we say that the previous row is the same row where we are
        
    
    if ( row == (rows-1)): # In case the row is the last one (the border) and we dont have a posterior row
        
        nrow=image[row,:,:] # The next row is going to be the actual row we are
        
    else:
        
        nrow=image[row+1,:,:]# If is the case where the row selected is not the last one then we can declare that the next row is just the following one
    
   
    frow = np.zeros((cols,depth)) # We declare which is going to be the future filtered row
    #frow = srow
    
    # Once prow, nrow, srow and frow are initialize we are going to go through every column and level of depth
    
    for d in range(depth): #first every depth 
        
        for c in range(cols): # of every depth d we go through every column
            
            if c > 0 and c < cols - 1: # In case that the column is in the middle and not any border
                
                # We are creating the matrix n of neighbors with the nine positions.  
                
                n = np.array([[prow[c-1,d], prow[c,d], prow[c+1,d]],[srow[c-1,d], srow[c,d], srow[c+1,d]],[nrow[c-1,d], nrow[c,d], nrow[c+1,d]]])
                
            elif c <= 0: # In case that the column is in the left border
                
                # We are creating the matrix n of neighbors with the nine positions, but in this case the columns from the left side are going to be equal to the actual column we are in 
                
                n = np.array([[prow[c,d], prow[c,d], prow[c+1,d]],[srow[c,d], srow[c,d], srow[c+1,d]],[nrow[c,d], nrow[c,d], nrow[c+1,d]]])
                
            elif c >= cols - 1 : # In case that the column is in the right border
                
                # We are creating the matrix n of neighbors with the nine positions, but in this case the columns from the right side are going to be equal to the actual column we are in.
                
                n = np.array([[prow[c-1,d], prow[c,d], prow[c,d]],[srow[c-1,d], srow[c,d], srow[c,d]],[nrow[c-1,d], nrow[c,d], nrow[c,d]]])
                             
                  
             #Multiplying the filter (my_filter) by the neighborhood matrix (n), every position of one matrix is going to be multiplied by the same position of the another matrix
            
            matriz = np.array([[n[0][0]*my_filter[0][0], n[0][1]*my_filter[0][1], n[0][2]*my_filter[0][2]],[n[1][0]*my_filter[1][0], n[1][1]*my_filter[1][1], n[1][2]*my_filter[1][2]], [n[2][0]*my_filter[2][0],n[2][1]*my_filter[2][1], n[2][2]*my_filter[2][2]]]) 
            
             #Modifying the pixel and putting the value of the sum of the neighborhood, because that is how the formula of filtering defends
                
            frow[c][d] = sum(sum(matriz))
    
    
    with shared_space.get_lock(): #Saving in the shared memory the filtered row
        
        shared_matrix[row, :, :] = frow
        # We do it at the end so the shared_matrix is modified at the end where all the loop and the conditionals have happened 
        
 



# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

