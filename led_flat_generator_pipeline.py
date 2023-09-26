#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:37:29 2023

@author: janmejoy

-Flat generator to be implemented in SUIT pipeline.
-Bias correction implemented using bias value from overscan regions.
-Under ideal condition, Master Dark subtraction should be done.
-Darks were not recorded for LED imaging sequence, as per instruction, due to 
tight time constraints during FBT.
"""

from astropy.convolution import convolve
from astropy.convolution import Box2DKernel
from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

def add(folder): #adds similar LED images in path 'folder' after bias correction for each quadrant.
#Bias correction is done by using the pixels in the overscan region.
    os.chdir(folder)
    files= sorted(os.listdir(folder))[0::2]
    data_sum=0
    for file in files:
        data=fits.open(file)[0].data
        data= data.astype(int)
        #efgh calculates the bias values from the overscan region of each quadrant
        e, f, g, h= np.mean(data[25:2068-25, 25:250-25]), np.mean(data[25:2068-25, -250+25:-25]), np.mean(data[2068+25:-25, 25:250-25]), np.mean(data[2068+25:-25, -250+25:-25])
        #bias correction is applied to each quadrant.
        data[:2068, :2352]=data[:2068, :2352]-e
        data[:2068, 2352:]=data[:2068, 2352:]-f
        data[2068:, :2352]= data[2068:, :2352]-g
        data[2068:, 2352:]= data[2068:, 2352:]-h
        data_sum= data_sum+data
    return (data_sum)

def blur(data, kernel): #blurring function
    return(convolve(data, Box2DKernel(kernel), normalize_kernel=True))

folder= '/run/media/janmejoy/data/essentials/projects/solar_physics/SUIT_FM_data/LED_data/2023-05-31_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005'
stacked_led_img= add(folder)
led_flat_field= stacked_led_img/blur(stacked_led_img, 11) #generates LED flat.
#Based on previous study, kernel size of 11x11 px is used for boxcar blurring in 'blur' function.

#Data visualization only.
plt.figure()
plt.subplot(1,2,1)
plt.imshow(stacked_led_img)
plt.colorbar()
plt.title("Bias corrected Master LED Image")
plt.subplot(1,2,2)
plt.imshow(led_flat_field, vmin=0.9, vmax=1.1)
plt.title("LED flat field")
plt.colorbar()
plt.show()


