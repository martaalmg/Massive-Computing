#!/usr/bin/env python
# coding: utf-8

# # MyFunctions2 module

# In[ ]:


import multiprocessing as mp
import numpy as np
from multiprocessing import Semaphore, Lock, Process, Pool
from multiprocessing.sharedctypes import Value, Array, RawArray
import ctypes


# In[ ]:


def withdraw(balance):     
    for _ in range(10000): 
        balance.value = balance.value - 1
  


# In[ ]:


def deposit(balance):     
    for _ in range(10000): 
        balance.value = balance.value + 1


# In[ ]:


def withdraw2(balance, lock):     
    for _ in range(10000): 
        lock.acquire() 
        balance.value = balance.value - 1
        lock.release() 


# In[ ]:


def deposit2(balance, lock):     
    for _ in range(10000): 
        lock.acquire() 
        balance.value = balance.value + 1
        lock.release() 


# In[ ]:

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

def init_channels(M,S,L,R):
    return

def decode_L(M,S, numprocess, lock):
    
    result_array = np.zeros(len(M))
    for i in range(M):

        result_array = M[i] + S[i]

    return result_array

def decode_R(M,S, numprocess, lock):
    
    result_array = np.zeros(len(M))    
    for i in range(M):
        result_array = M[i] - S[i]
    
    return result_array



#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

