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
-Made for use with ground FBT data.

-2023-10-27: PRNU % calculation for each quadrant and full flat field image is introduced
as a function

-2023-11-17: 255 nm LED data to be analysed. Function added.

-2023-11-23: Function added to analyse the performace of FF correction on single 
255 nm and 355 nm images.

-2023-11-24: Single flat generator function is made for 355 and 255 nm LED images.
Previously two separate functions were present. calib_stat crop areas updated.
"""

import glob
import datetime
from astropy.convolution import convolve
from astropy.convolution import Box2DKernel
from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

def add(files): #adds similar LED images in path 'folder' after bias correction for each quadrant.
#Bias correction is done by using the pixels in the overscan region.
    data_sum=0
    for file in files:
        hdu= fits.open(file)[0]
        gain_e, gain_f= float(hdu.header['AMP_G_E']), float(hdu.header['AMP_G_F'])
        gain_g, gain_h= float(hdu.header['AMP_G_G']), float(hdu.header['AMP_G_H'])
        data=hdu.data
        data= data.astype(int)
        #efgh calculates the bias values from the overscan region of each quadrant
        e, f, g, h= np.mean(data[25:2068-25, 25:250-25]), np.mean(data[25:2068-25, -250+25:-25]), np.mean(data[2068+25:-25, 25:250-25]), np.mean(data[2068+25:-25, -250+25:-25])
        #bias correction is applied to each quadrant.
        data[:2068, :2352]=(data[:2068, :2352]-e)*gain_e
        data[2068:, :2352]= (data[2068:, :2352]-g)*gain_f
        data[:2068, 2352:]= (data[:2068, 2352:]-f)*gain_g
        data[2068:, 2352:]= (data[2068:, 2352:]-h)*gain_h
        data_sum= (data_sum+data)/np.mean([gain_e, gain_f,gain_g, gain_h])
    return (data_sum)

def blur(data, kernel): #blurring function
    return(convolve(data, Box2DKernel(kernel), normalize_kernel=True))

def prnu_generator(folder, kernel, name): #running for 355nm LED
    stacked_led_img= add(folder)
    led_flat_field= stacked_led_img/blur(stacked_led_img, kernel) #generates LED flat.
    #Based on previous study, kernel size of 11x11 px is used for boxcar blurring in 'blur' function.
    prnu_measure(led_flat_field) #for printing PRNU values.
    return(led_flat_field)

def prnu_measure(data): #used to find % pixel response non uniformity (PRNU) in each quad and full image
    e,f,g,h= data[7:2068, 309:2352], data[7:2068, 2352:4390], data[2068:4127, 309:2352], data[2068:4127, 2352:4390]
    stdev= np.array([np.std(e), np.std(f), np.std(g), np.std(h)])
    mean= np.array([np.mean(e), np.mean(f), np.mean(g), np.mean(f)])
    prnu_quadrants= stdev*100/mean #e,f,g,h        
    prnu_full_frame= np.std((data[7:4127,309:4390])*100/np.mean(data[7:4127,309:4390]))
    print('PRNU_full_frame %= ', prnu_full_frame)
    print('PRNU % [e,f,g,h]= ',prnu_quadrants)

#finds degree of correction implemented by PRNU correction.
def calib_stats(single, corrected, prnu, croprow, cropcol, size):
    crop_raw= single[croprow:croprow+size, cropcol:cropcol+size]
    crop_cor= corrected[croprow:croprow+size, cropcol:cropcol+size]
    crop_prnu= prnu[croprow:croprow+size, cropcol:cropcol+size]
    sd_raw= np.std(crop_raw)/np.mean(crop_raw)
    sd_cor= np.std(crop_cor)/np.mean(crop_cor)
    sd_prnu= np.std(crop_prnu)/np.mean(crop_prnu)
    calc_cor= np.sqrt(np.abs(sd_raw**2-sd_prnu**2))
    print("[1] Sdev Raw:", sd_raw)
    print("[2] Sdev Corrected:", sd_cor)
    print("[3] Sdev PRNU: ", sd_prnu)
    print("[4] sqrt(Raw^2-PRNU^2): ", calc_cor)
    print("Upon good FF correction, [2] and [4] should match well")
    print("Poission Noise (ADU)", np.sqrt(np.mean(crop_cor))/np.mean(crop_cor))

def lighten(image_list):
    '''
    Lighten blend
    image_list= List of 2D numpy arrays
    '''
    lighten_blend=np.zeros(np.shape(image_list[0]))
    for image in image_list:
        for i in range(4096):
            for j in range(4096):
                if image[i,j] > lighten_blend[i,j]:
                    lighten_blend[i,j]=image[i,j]
    return(lighten_blend)

def prep_header(fname, mfg, data_date):
    header=fits.Header()
    header['VERSION']=('beta', 'Version name for the Flat Field')
    header['FNAME']=(fname, 'LED ID for SUIT')
    header['MFG_DATE']=(mfg, 'Manufacturing date for the FITS file')
    header['DATADATE']=(data_date,'Date of raw data recording')
    return (header)

if __name__=='__main__':
    ### Flatfield correction on LED images ###
    project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/flat_field/LED/ground_PRNU_project/')
    filelist= glob.glob(project_path+"data/raw/LED*")
    save= False
    sav= os.path.join(project_path, 'products/')
    mfg= str(datetime.date.today()) #manufacturing date
    
    kernel_355= 11 #default is 11 for PRNU
    kernel_255= 13 # default is 13 for PRNU
    
    aa_255, ff_255=[], [] #aa and ff represent different sets of 4 LEDs.
    aa_355, ff_355=[], []
    
    for file in filelist: #segregating files based on LEDONOFF ID
        ledstat=fits.open(file)[0].header['LEDONOFF']
        if (ledstat=='55'):
            print (ledstat, fits.open(file)[0].header['FW1POS'], file)
            ff_255.append(file)
        elif(ledstat=='aa'):
            print (ledstat, fits.open(file)[0].header['FW1POS'], file)
            aa_255.append(file)
        elif(ledstat=='5500'):
            print (ledstat, fits.open(file)[0].header['FW1POS'], file)
            ff_355.append(file)
        elif(ledstat=='aa00'):
            print (ledstat, fits.open(file)[0].header['FW1POS'], file)
            aa_355.append(file)
            
            
    ### 355 nm ###
    prnu_355_aa= prnu_generator(aa_355, kernel_355, "PRNU 355 nm")
    prnu_355_ff= prnu_generator(ff_355, kernel_355, "PRNU 355 nm")
    prnu_355_common= lighten([prnu_355_ff, prnu_355_aa])
    sav_hdu= fits.PrimaryHDU(prnu_355_common, header= prep_header('prnu_355_common.fits', mfg, '2023-05-31'))
    if save: sav_hdu.writeto(f'{sav}_2023-05-31_prnu_355_common.fits')
    
    ### 255 nm ###
    prnu_255_aa= prnu_generator(aa_255, kernel_255, "PRNU 255 nm")
    prnu_255_ff= prnu_generator(ff_255, kernel_255, "PRNU 255 nm")
    prnu_255_common= lighten([prnu_255_ff, prnu_255_aa])
    sav_hdu= fits.PrimaryHDU(prnu_355_common, header= prep_header('prnu_255_common.fits', mfg, '2023-06-01'))
    if save: sav_hdu.writeto(f'{sav}_2023-06-01_prnu_255_common.fits')
    
'''
    
    single_355=fits.open(folder+'/LED_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005_000001.fits')[0].data
    single_355=single_355.astype(int)
    corrected_355= single_355/prnu_355
    print('\n**** 355 nm flat stats ****')
    calib_stats(single_355, corrected_355, prnu_355, 2500, 2500, 25)
    
    
    ### 255 nm ###
    folder= os.path.join(project_path, 'data/raw/2023-06-01_255_4_led_10_times_fw1_07_fw2_06_6s_004')
    prnu_255= flat_generator(folder, 13, "PRNU 255 nm")
    single_255=fits.open(folder+'/LED_255_4_led_10_times_fw1_07_fw2_06_6s_004_000001.fits')[0].data
    single_255=single_255.astype(int)
    corrected_255= single_255/prnu_255
    print('\n**** 255 nm flat stats ****')
    calib_stats(single_255, corrected_255, prnu_255, 1000, 2500, 25)

'''