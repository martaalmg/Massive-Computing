#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
# Due a bug in the multiprocessing module implemented for Windows Operating System, the functions which will be executed in the parallel threads MUST be implemented in a separated file, and import them in the main programs.
# 
# In order to be loaded in you own program, you have to write your own functions here, and export to a python ".py" file, to be imported in  the main script.
# 
# To export to a python file, select in the *File*  menu, the option *Download as* and save as *Python .py* file.

# ## Functions needed for FirstParallel notebook

import numpy as np

def init_second(my_matrix2):
    global matrix_2
    matrix_2=my_matrix2
    print(matrix_2.shape)


# In[ ]:


def init_globalimage(img,filt):
    global image
    global my_filter
    image=img
    my_filter=filt


# In[ ]:


def parallel_matmul(v):
    # v: is the input row
    # matrix_2: is the second matrix, shared by memory
    
    #here we calculate the shape of the second matrix, to generate the resultant row
    matrix_2 # we will uses the global matrix
    
    (rows,columns)=matrix_2.shape
    
    #we allocate the final vector of size the number of columns of matrix_2
    d=np.zeros(columns)
    
    #we calculate the dot product between vector v and each column of matrix_2
    for i in range(columns):
        d[i] = np.dot(v,matrix_2[:,i])
    
    #returns the final vector d
    return d


# In[ ]:


def parallel_filtering_image(r): # The input is the image row to filter (r)
    
    
    global image # image is the global memory array
    global my_filter # my_filter is the filter shape to apply to the image
    
   
    (rows,cols,depth) = image.shape # from the global variaable, gets the image shape
    
    srow=image[r,:,:] # fetch the r row from the original image
    
    # Now it starts to declare the variables previous row (prow) and the next row (nrow)
    
    if ( r > 0 ): # In case the row is not the first (in the border)
        prow = image[r-1,:,:] # We just declare that the previous row could be perfectly the before one          
    else:
        prow=image[r,:,:] # If that is not the case and it is the case where is the first row then the previous row does not exist and we say that the previous row is the same row where we are
        
    if ( r == (rows-1)): # In case the row is the last one (the border) and we dont have a posterior row
        nrow=image[r,:,:] # The next row is going to be the actual row we are
    else:
        nrow=image[r+1,:,:] # If is the case where the row selected is not the last one then we can declare that the next row is just the following one
   
    frow = srow # Initialize the future filtered row 
    
    # Another way of doing it ^^^ could be : frow = np.zeros(cols, depth)

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
                                    
    
    return frow # The output is the filtered row 
    

# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

