#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:37:29 2023

@author: janmejoy

Flat generator to be implemented in SUIT pipeline.
"""

from astropy.convolution import convolve
from astropy.convolution import Box2DKernel
from astropy.io import fits
import os

def blur(data, kernel): #blurring function
    return(convolve(data, Box2DKernel(kernel), normalize_kernel=True))

def flat_gen(file): #flat field generator
    led= fits.open(file)[0].data #opens reduced and stacked LED image
    return(led/blur(led, 11))


folder= '/run/media/janmejoy/data/essentials/projects/solar_physics/SUIT_FM_data/LED_data/2023-05-31_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005'
file= 'LED_4_led_355nm_p75_current_10_times_100ms_fw1_04_fw2_07_005_000001.fits'
os.chdir(folder)

led_flat= flat_gen(file) #generates LED flat. 
#ensure that the file being fed it a reduced and stacked LED flat. 
#Prefer an average of >20 LED images
