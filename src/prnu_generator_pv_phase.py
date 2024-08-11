#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:48:51 2023
-Ran on SUIT server
-LED PRNU model generator from PV phase data.
-Same functions used for PRNU calculations as that used for FBT data.
-Level 1 data is used for analysis here.
-Bias correction is not used here.
-4LED Files are segregated based on image headers
-Last modified:
    2023-11-28
    2023-12-18: Added write fits option.
    2024-02-21: Added single point user defined entry for save location 
    and kernel size.
    Added minor comments. Made dustmap with 25 px kernel.
-Added feature for saving PRNU images as per the raw image date.
- 2024-07-24: Added Level 1 conversion of ground images makes the images hori-
zontally flipped, compared to onboard LED images. A fliplr has been added.
- 2024-08-09: Module added to remove quadrant jump lines. Medians across 15 px
taken for each row in the quadrant change regions.
- 2024-08-10: Module added to remove dust spots on PRNU image.
- 2024-08-11: Added Dict for data recording dates. Renamed data date variable in
flat generator function.

@author: janmejoy
"""

import numpy as np
import os
import glob
from astropy.io import fits
from astropy.convolution import convolve
from astropy.convolution import Box2DKernel
import datetime
from skimage.morphology import dilation, disk, erosion


def add(filelist):
    data_sum=0
    for file in filelist:
        data_sum=data_sum+fits.open(file)[0].data
    return(data_sum)

def blur(data, kernel): #blurring function
    return(convolve(data, Box2DKernel(kernel), normalize_kernel=True))

def prep_header(fname, mfg, data_date):
    header=fits.Header()
    header['VERSION']=('beta', 'Version name for the Flat Field')
    header['FNAME']=(fname, 'LED ID for SUIT')
    header['MFG_DATE']=(mfg, 'Manufacturing date for the FITS file')
    header['DATADATE']=(data_date,'Date of raw data recording')
    return (header)

def flat_generator(filelist, kernel, name, data_date, save=None):
    stacked_led_img= add(filelist)
    led_flat_field= stacked_led_img/blur(stacked_led_img, kernel) #generates LED flat.
    led_flat_field= np.fliplr(led_flat_field)
    hdu=fits.PrimaryHDU(led_flat_field, header=prep_header(name, mfg, data_date))
    if save==True: hdu.writeto(f'{sav}{name}.fits', overwrite=True)
    #Based on previous study, kernel size of 11x11 px is used for boxcar blurring in 'blur' function.
    return(led_flat_field)

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

#finds degree of correction implemented by PRNU correction.
def calib_stats(single_filename, prnu_file, croprow, cropcol, size, name):
    single= fits.open(single_filename)[0].data #convert to photoelectrons
    corrected= single/prnu_file
    crop_raw= single[croprow:croprow+size, cropcol:cropcol+size]
    crop_cor= corrected[croprow:croprow+size, cropcol:cropcol+size]
    crop_prnu= prnu_file[croprow:croprow+size, cropcol:cropcol+size]
    
    sd_raw= np.std(crop_raw)
    sd_cor= np.std(crop_cor)
    sd_prnu= np.std(crop_prnu)
    calc_cor= np.sqrt(np.abs(sd_raw**2-sd_prnu**2))
    ### PERCENT ERRORS ###
    sd_raw_pc= sd_raw*100/np.mean(crop_raw)
    sd_cor_pc= sd_cor*100/np.mean(crop_cor)
    sd_prnu_pc= sd_prnu*100/np.mean(crop_prnu)
    calc_cor_pc= calc_cor*100/np.mean(crop_cor)
    
    print(f"\n{name}")
    print("[1] Sdev Raw:", sd_raw_pc)
    print("[2] Sdev Corrected:", sd_cor_pc)
    print("[3] Sdev PRNU: ", sd_prnu_pc)
    print("[4] sqrt(Raw^2-PRNU^2): ", calc_cor_pc)
    print("Avg counts=", np.mean(crop_raw))
    print("Upon good FF correction, [2] and [4] should match well \n")
    print("Mean", np.mean(crop_cor))
    
def remove_center_line(data, s):
    '''
    Parameters
    ----------
    data : 2D numpy array
        Processed PRNU data.
    s : integer
        size of center filter.

    Returns
    -------
    Filtered image np array
    '''
    for i in range(4096):
        data[i, 2048-s: 2048+s]= np.mean(data[i, 2048-s: 2048+s])
        data[2048-s: 2048+s, i]= np.mean(data[2048-s: 2048+s, i])
    return(data)

def remove_dust(data, plot=None, sav=None):
    particle_mask= dilation(data<0.97, disk(3))
    filtered=dilation(erosion(data, disk(1)), disk(6))*particle_mask
    filtered_image= (data*np.logical_not(particle_mask))+filtered
    spike_mask= erosion(filtered_image<1.02, disk(3))
    spike_filtered=erosion(filtered_image, disk(8))*np.invert(spike_mask)
    spike_filtered_image= (filtered_image*spike_mask)+spike_filtered
    return(spike_filtered_image)

if __name__=='__main__':
    project_path= os.path.expanduser('~/Dropbox/Janmejoy_SUIT_Dropbox/flat_field/LED/ground_PRNU_project/')
    filelist= glob.glob(project_path+"data/processed/*")
    save= True
    sav= os.path.join(project_path, 'products/')
    mfg= str(datetime.date.today()) #manufacturing date
    date_dict= {'255':'2023-06-01', '355':'2023-05-31'} #data dates
    
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
        
    ## Make PRNU for 255 ##
    date= date_dict['355'] #data date
    print(f'{date}_prnu_355_ff')
    prnu_355_ff = flat_generator(ff_355, kernel_355, f'{date}_prnu_355_ff', date, save)
    
    print(f'{date}_prnu_355_aa')
    prnu_355_aa = flat_generator(aa_355, kernel_355, f'{date}_prnu_355_aa', date, save)
    
    prnu_355_common= lighten([prnu_355_ff, prnu_355_aa])
    prnu_355_common= remove_center_line(prnu_355_common, 5)
    prnu_355_common= remove_dust(prnu_355_common)
    sav_hdu= fits.PrimaryHDU(prnu_355_common, header= prep_header('prnu_355_common.fits', mfg, date))
    if save: sav_hdu.writeto(f'{sav}{date}_prnu_355_common.fits', overwrite=True)
    calib_stats(aa_355[0], prnu_355_common, 2200, 1500, 25, f'{date}_prnu_355_aa') #2200, 1500, 25,
    calib_stats(ff_355[0], prnu_355_common, 2200, 1500, 25, f'{date}_prnu_355_ff')

    ## Make PRNU for 255 ##
    date=date_dict['255'] #data date
    print(f'{date}_prnu_255_ff')
    prnu_255_ff = flat_generator(ff_255, kernel_255, f'{date}_prnu_255_ff', date, save)
    
    print(f'{date}_prnu_255_aa')
    prnu_255_aa = flat_generator(aa_255, kernel_255, f'{date}_prnu_255_aa', date, save)
    
    prnu_255_common= lighten([prnu_255_ff, prnu_255_aa])
    prnu_255_common= remove_center_line(prnu_255_common, 5)
    prnu_255_common= remove_dust(prnu_255_common)
    sav_hdu= fits.PrimaryHDU(prnu_255_common, header= prep_header('prnu_255_common.fits', mfg, date))
    if save: sav_hdu.writeto(f'{sav}{date}_prnu_255_common.fits', overwrite=True)
    calib_stats(ff_255[0], prnu_255_common, 2000, 3800, 25, f'{date}_prnu_255_ff')
    calib_stats(aa_255[0], prnu_255_common, 2000, 3800, 25, f'{date}_prnu_255_aa')


    
    
