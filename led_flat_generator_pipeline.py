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

-2023-10-27: PRNU % calculation for each quadrant and full flat field image is introduced
as a function

-2023-11-17: 255 nm LED data to be analysed. Function added.
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
    files= sorted(os.listdir(folder))[::2]
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

def prnu_measure(data): #used to find % pixel response non uniformity (PRNU) in each quad and full image
        e,f,g,h= data[7:2068, 309:2352], data[7:2068, 2352:4390], data[2068:4127, 309:2352], data[2068:4127, 2352:4390]
        stdev= np.array([np.std(e), np.std(f), np.std(g), np.std(h)])
        mean= np.array([np.mean(e), np.mean(f), np.mean(g), np.mean(f)])
        prnu_quadrants= stdev*100/mean #e,f,g,h        
        prnu_full_frame= np.std((data[7:4127,309:4390])*100/np.mean(data[7:4127,309:4390]))
        print('PRNU_full_frame %= ', prnu_full_frame)
        print('PRNU % [e,f,g,h]= ',prnu_quadrants)

def nm355(): #running for 355nm LED
    folder= '/home/janmejoy/Dropbox/Janmejoy_SUIT_Dropbox/flat_field/LED_data/2023-05-31_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005'
    stacked_led_img= add(folder)
    led_flat_field= stacked_led_img/blur(stacked_led_img, 11) #generates LED flat.
    #Based on previous study, kernel size of 11x11 px is used for boxcar blurring in 'blur' function.
    
    #Data visualization only.
    plt.figure("PRNU_355 nm")
    plt.subplot(1,2,1)
    plt.imshow(stacked_led_img)
    plt.colorbar()
    plt.title("Bias corrected Master LED Image")
    plt.subplot(1,2,2)
    plt.imshow(led_flat_field, vmin=0.97, vmax=1.03)
    plt.title("PRNU_355 nm")
    plt.colorbar()
    plt.show()
    
    prnu_measure(led_flat_field) #for printing PRNU values.
    return(led_flat_field)
    
def nm255(): #running for 255nm LED
    folder= '/home/janmejoy/Dropbox/Janmejoy_SUIT_Dropbox/flat_field/LED_data/2023-06-01_255_4_led_10_times_fw1_07_fw2_06_6s_004'
    stacked_led_img= add(folder)
    led_flat_field= stacked_led_img/blur(stacked_led_img, 11) #generates LED flat.
    #Based on previous study, kernel size of 11x11 px is used for boxcar blurring in 'blur' function.
    
    #Data visualization only.
    plt.figure("PRNU_255 nm")
    plt.subplot(1,2,1)
    plt.imshow(stacked_led_img)
    plt.colorbar()
    plt.title("Bias corrected Master LED Image")
    plt.subplot(1,2,2)
    plt.imshow(led_flat_field, vmin=0.97, vmax=1.03)
    plt.title("PRNU_255 nm")
    plt.colorbar()
    plt.show()
    
    prnu_measure(led_flat_field) #for printing PRNU values.
    return(led_flat_field)

def residual(crop_raw, crop):
    sig_raw= np.std(crop_raw)
    mean_raw= np.mean(crop_raw)
    sig_corr= np.std(crop)
    mean_corr=np.mean(crop)
    res_error= np.sqrt((sig_raw**2-sig_corr**2))/mean_corr
    print(sig_raw, mean_raw, sig_corr, mean_corr, res_error)
    
prnu_355= nm355()
prnu_255= nm255()

single_255=fits.open('/home/janmejoy/Dropbox/Janmejoy_SUIT_Dropbox/flat_field/LED_data/2023-06-01_255_4_led_10_times_fw1_07_fw2_06_6s_004/LED_255_4_led_10_times_fw1_07_fw2_06_6s_004_000002.fits')[0].data
single_255=single_255.astype(int)
corrected_255= single_255/prnu_255
residual(single_255[2500:2600, 2500:2600], corrected_255[2500:2600, 2500:2600])
residual(single_255[1500:1600, 1500:1600], corrected_255[1500:1600, 1500:1600])


single_355=fits.open('/home/janmejoy/Dropbox/Janmejoy_SUIT_Dropbox/flat_field/LED_data/2023-05-31_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005/LED_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005_000002.fits')[0].data
single_355=single_355.astype(int)
corrected_355= single_355/prnu_355
residual(single_355[2500:2600, 2500:2600], corrected_355[2500:2600, 2500:2600])
residual(single_355[1100:1200, 1100:1200], corrected_355[1100:1200, 1100:1200])
