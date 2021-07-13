# Cyna Shirazinejad, Drubin/Barnes Lab, last modified 9/23/20
# 
# tools for displaying track attributes from cmeAnalysis

import os
import sys
sys.path.append(os.getcwd())
from generate_index_dictionary import return_index_dictionary
from return_track_attributes import (return_track_lifetime, 
                                     return_track_amplitude_one_channel, 
                                     return_puncta_x_position_whole_track,
                                     return_puncta_y_position_whole_track,
                                     return_distance_traveled_from_origin,
                                     return_distance_between_two_channel,
                                     return_track_amplitude,
                                     return_track_category,
                                     return_is_CCP,
                                     return_frames_in_track,
                                     return_track_x_position,
                                     return_track_amplitude_no_buffer_channel)
import alignment as alignment_tools
import return_track_attributes                                   
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
import warnings
import matplotlib
import itk
import imageio
index_dictionary = return_index_dictionary()
from IPython.display import display
import random
import scipy.interpolate as interpolate
import seaborn as sns
import alignment

 
def plot_separated_cohorts(axes, 
                           vectors, 
                           alignment_channel, 
                           cohort_bounds, 
                           indices_first_axis, 
                           indices_second_axis, 
                           labels,
                           colors,
                           line_cutoff_index=[],
                           line_cutoff_regions=[],
                           horizontal_shift_index=[],
                           horizontal_shift=[],
                           line_removal_index=[],
                           line_kept_regions=[],
                           framerate=1,
                           norm_intensity=False):
    
    num_ticks = 0
    x_labels = []
    for cohort in cohort_bounds:

        x_labels+=list(range(0,cohort[1]-10+1,10))

        num_ticks += int(cohort[1]/10)+1

    offsets = []
    
    cohort_samples = cohort_bounds.copy()
    
    cohort_samples[-1] = [cohort_samples[-1][0], cohort_samples[-1][1]+1]
    
    for i in range(len(cohort_samples)):
        
        if i==0:
            offsets.append(0)
        else:
            offsets.append(cohort_samples[i][0])

        if i > 0:

            offsets[i]+=offsets[i-1]
            
            
    num_stds = 0.25
    num_in_cohort = []
    max_cohort_indices = []
    
    amplitudes = np.array(vectors, dtype='object')[:,indices_first_axis]
    print(offsets)
    for i in range(1,len(offsets)):

        plt.axvline(offsets[i], 0, 1, linestyle='--', color='black', alpha=0.9)

    for offset_index, cohort in enumerate(cohort_samples):

        class_intensities = []

        cohort_temp_class = []

        for i in range(len(amplitudes)):

            lifetime = len(amplitudes[i][0])*framerate

            if lifetime >= cohort[0] and lifetime < cohort[1]:

                cohort_temp_class.append(vectors[i])

        num_in_cohort.append(len(cohort_temp_class))
        frames_before_alignment = []
        frames_after_alignment = []

        if cohort_temp_class!=[]:
        
            for track_index in range(len(cohort_temp_class)):

                frames_before_alignment.append(np.argmax(cohort_temp_class[track_index][alignment_channel]))
                frames_after_alignment.append(len(cohort_temp_class[track_index][alignment_channel]) - np.argmax(cohort_temp_class[track_index][alignment_channel]) - 1)

            padding_amount = np.max([np.max(frames_before_alignment), np.max(frames_after_alignment)])

            cohort_temp_class = alignment.return_shifted_amplitudes(cohort_temp_class, 
                                                                    frames_before_alignment, 
                                                                    frames_after_alignment, 
                                                                    padding_amount, 
                                                                    True)

            average_cohort_class = np.nan_to_num(np.nanmean(cohort_temp_class,axis=0,dtype=np.float64)) # calculate average and std of intensity in class cohort
            std_cohort_class = num_stds*np.nan_to_num(np.nanstd(cohort_temp_class,axis=0,dtype=np.float64))
            

            
            for i in indices_first_axis:

                ch_current = average_cohort_class[i,:]

                t, c, k = interpolate.splrep(np.arange(len(ch_current)), ch_current, k=3)

                spline = interpolate.BSpline(t, c, k, extrapolate=False)

                splined_amps = spline(np.linspace(0, len(ch_current), cohort_bounds[offset_index][1]))

                if offset_index in line_removal_index:
                
                    splined_amps = splined_amps[line_kept_regions[offset_index][0]:line_kept_regions[offset_index][1]]
                    
                avg_intensities = splined_amps.copy()
                
                if i==alignment_channel:

                    max_cohort_indices.append(np.nanargmax(splined_amps))

                if offset_index != 0:

                    labels = [None]*average_cohort_class.shape[0]

                if i in line_cutoff_index:

                    for regions in line_cutoff_regions[i]:

                        splined_amps[regions[0]:regions[1]] = np.NaN

                if i in horizontal_shift_index:

                    x_axis = horizontal_shift[i]+np.linspace(0+offsets[offset_index],cohort_bounds[offset_index][1]+offsets[offset_index],cohort_bounds[offset_index][1])

                else:

                    x_axis = np.linspace(0+offsets[offset_index],cohort_bounds[offset_index][1]+offsets[offset_index],cohort_bounds[offset_index][1])
                    
                if offset_index in line_removal_index:
                
                    x_axis = x_axis[line_kept_regions[offset_index][0]:line_kept_regions[offset_index][1]]
                
                if norm_intensity:
                    norm_factor = 1/np.nanmax(splined_amps)
                else:
                    norm_factor = 1
                axes[0].plot(x_axis, 
                         norm_factor*splined_amps,
                         colors[i], 
                         label=labels[i])

                ch_current = std_cohort_class[i,:]

                t, c, k = interpolate.splrep(np.arange(len(ch_current)), ch_current, k=3)

                spline = interpolate.BSpline(t, c, k, extrapolate=False)

                splined_amps = spline(np.linspace(0, len(ch_current), cohort_bounds[offset_index][1]))
                
                if offset_index in line_removal_index:
                
                    splined_amps = splined_amps[line_kept_regions[offset_index][0]:line_kept_regions[offset_index][1]]
                    
                if i in line_cutoff_index:

                    for regions in line_cutoff_regions[i]:

                        splined_amps[regions[0]:regions[1]] = np.NaN
                        avg_intensities[regions[0]:regions[1]] = np.NaN

                axes[0].fill_between(x_axis, 
                                 norm_factor*avg_intensities-norm_factor*splined_amps, 
                                 norm_factor*avg_intensities+norm_factor*splined_amps, 
                                 color=colors[i], 
                                 alpha=0.2)       



            for i in indices_second_axis:

                ch_current = average_cohort_class[i,:]

                t, c, k = interpolate.splrep(np.arange(len(ch_current)), ch_current, k=3)

                spline = interpolate.BSpline(t, c, k, extrapolate=False)

                splined_amps = spline(np.linspace(0, len(ch_current), cohort_bounds[offset_index][1]))

                if offset_index in line_removal_index:
                
                    splined_amps = splined_amps[line_kept_regions[offset_index][0]:line_kept_regions[offset_index][1]]
                    
                avg_intensities = splined_amps.copy()

                if offset_index != 0:

                    labels = [None]*average_cohort_class.shape[0]

                if i in line_cutoff_index:

                    for regions in line_cutoff_regions[i]:

                        splined_amps[regions[0]:regions[1]] = np.NaN

                if i in horizontal_shift_index:

                    x_axis = horizontal_shift[i]+np.linspace(0+offsets[offset_index],cohort_bounds[offset_index][1]+offsets[offset_index],cohort_bounds[offset_index][1])

                else:

                    x_axis = np.linspace(0+offsets[offset_index],cohort_bounds[offset_index][1]+offsets[offset_index],cohort_bounds[offset_index][1])
                
                if offset_index in line_removal_index:
                
                    x_axis = x_axis[line_kept_regions[offset_index][0]:line_kept_regions[offset_index][1]]
                
                
                axes[1].plot(x_axis, 
                         splined_amps,
                         colors[i], 
                         label=labels[i],
                         linestyle='--')

                ch_current = std_cohort_class[i,:]

                t, c, k = interpolate.splrep(np.arange(len(ch_current)), ch_current, k=3)

                spline = interpolate.BSpline(t, c, k, extrapolate=False)

                splined_amps = spline(np.linspace(0, len(ch_current), cohort_bounds[offset_index][1]))

                if offset_index in line_removal_index:
                
                    splined_amps = splined_amps[line_kept_regions[offset_index][0]:line_kept_regions[offset_index][1]]
                    
                
                if i in line_cutoff_index:

                    for regions in line_cutoff_regions[i]:

                        splined_amps[regions[0]:regions[1]] = np.NaN
                        avg_intensities[regions[0]:regions[1]] = np.NaN

                axes[1].fill_between(x_axis, 
                                 avg_intensities-splined_amps, 
                                 avg_intensities+splined_amps, 
                                 color=colors[i], 
                                 alpha=0.2) 
            
    return offsets, num_in_cohort, max_cohort_indices


def plot_cohorts_centered_on_aligned_channel(axes, 
                                             vectors, 
                                             alignment_channel, 
                                             cohort_bounds, 
                                             indices_first_axis, 
                                             indices_second_axis, 
                                             labels,
                                             colors,
                                             line_cutoff_index=[],
                                             line_cutoff_regions=[],
                                             horizontal_shift_index=[],
                                             horizontal_shift=[]):

    
    num_ticks = 0
    x_labels = []
    for cohort in cohort_bounds:

        x_labels+=list(range(0,cohort[1]-10+1,10))

        num_ticks += int(cohort[1]/10)+1

    offsets = []
    
    cohort_samples = cohort_bounds.copy()
    
    cohort_samples[-1] = [cohort_samples[-1][0], cohort_samples[-1][1]+1]
    
    for i in range(len(cohort_samples)):
        
        if i==0:
            offsets.append(0)
        else:
            offsets.append(cohort_samples[i][0])

        if i > 0:

            offsets[i]+=offsets[i-1]
            
            
    num_stds = 0.25
    num_in_cohort = []
    
    amplitudes = np.array(vectors, dtype='object')[:,indices_first_axis]



    for offset_index, cohort in enumerate(cohort_samples):

        class_intensities = []

        cohort_temp_class = []

        for i in range(len(amplitudes)):

            lifetime = len(amplitudes[i][0])

            if lifetime >= cohort[0] and lifetime < cohort[1]:

                cohort_temp_class.append(vectors[i])

        num_in_cohort.append(len(cohort_temp_class))
        frames_before_alignment = []
        frames_after_alignment = []

        for track_index in range(len(cohort_temp_class)):

            frames_before_alignment.append(np.argmax(cohort_temp_class[track_index][alignment_channel]))
            frames_after_alignment.append(len(cohort_temp_class[track_index][alignment_channel]) - np.argmax(cohort_temp_class[track_index][alignment_channel]) - 1)

        padding_amount = np.max([np.max(frames_before_alignment), np.max(frames_after_alignment)])

        cohort_temp_class = alignment.return_shifted_amplitudes(cohort_temp_class, 
                                                                frames_before_alignment, 
                                                                frames_after_alignment, 
                                                                padding_amount, 
                                                                True)

        average_cohort_class = np.nan_to_num(np.nanmean(cohort_temp_class,axis=0,dtype=np.float64)) # calculate average and std of intensity in class cohort
        std_cohort_class = num_stds*np.nan_to_num(np.nanstd(cohort_temp_class,axis=0,dtype=np.float64))
        
        ch_align = average_cohort_class[alignment_channel, :]
        
        t, c, k = interpolate.splrep(np.arange(len(ch_align)), ch_align, k=3)

        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        splined_amps = spline(np.linspace(0, len(ch_align), cohort_bounds[offset_index][1])) 
        
        horizontal_alignment = np.nanargmax(splined_amps)
        
        for i in indices_first_axis:
            
            ch_current = average_cohort_class[i,:]
            
            t, c, k = interpolate.splrep(np.arange(len(ch_current)), ch_current, k=3)
            
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            
            splined_amps = spline(np.linspace(0, len(ch_current), cohort_bounds[offset_index][1]))
            
            avg_intensities = splined_amps.copy()

            if offset_index != 0:
                
                labels = [None]*average_cohort_class.shape[0]
            
            if i in line_cutoff_index:
                
                for regions in line_cutoff_regions[i]:
                    
                    splined_amps[regions[0]:regions[1]] = np.NaN
            
             
            x_axis = -horizontal_alignment + np.linspace(0,cohort_bounds[offset_index][1],cohort_bounds[offset_index][1])
                
            axes[0].plot(x_axis, 
                     splined_amps,
                     colors[i], 
                     label=labels[i])

            ch_current = std_cohort_class[i,:]
            
            t, c, k = interpolate.splrep(np.arange(len(ch_current)), ch_current, k=3)
            
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            
            splined_amps = spline(np.linspace(0, len(ch_current), cohort_bounds[offset_index][1]))
            
            if i in line_cutoff_index:
                
                for regions in line_cutoff_regions[i]:
                    
                    splined_amps[regions[0]:regions[1]] = np.NaN
                    avg_intensities[regions[0]:regions[1]] = np.NaN
                    
            axes[0].fill_between(x_axis, 
                             avg_intensities-splined_amps, 
                             avg_intensities+splined_amps, 
                             color=colors[i], 
                             alpha=0.2)  

def fit_cohorts(tracks, 
                class_indices, 
                figsize, 
                plot_shape, 
                num_stds = 0.25, 
                cohorts=[[0,20],[21,40],[41,60],[61,80],[81,100],[101,120],[121,140],[141,160],[161,180],[181,200],[201,220],[221,240]], 
                filename = '',
                alignment='interpolate'):
    """
    Display lifetime-binned cohorts of tracks separated by different class labels. 
    Tracks in each cohort are interpolated to the length of the ceiling of the cohort.
    
    Args:
        tracks (ndarray): numpy array of cmeAnalysis' ProcessedTracks.mat 
        class_indices (list of lists): each internal list contains indices for tracks in a designated class
        figsize (tuple): figure size for all subplots
        plot_shape (tuple): size of each subplot
        num_stds (float, optional): number of standard deviations of intensity for plotting intensity traces
        cohorts (list of lists, optional): each internal list contains the upper and lower bounds of lifetimes for each cohort
        filename (string, optional): location for plots to be saved
        interpolate (string, optional): method for averaging intensities in cohorts
        
    Returns:
        None
        
    """
    print('fitting to the following cohorts: ' + str(cohorts))
    print()
    f = figure(dpi=50,figsize=figsize)
    subplot_index = 1
    warnings.filterwarnings("ignore")

    class_intensities = []
    
    for i,class_tmp in enumerate(class_indices): # iterate through each class of labels
    
        class_intensities_current = []
        
        ax = f.add_subplot(plot_shape[0], plot_shape[1], subplot_index)
        ax.set_title('class: ' + str(i))
        
        for cohort in cohorts: # gather intensities for each cohort for all members of current class
#             print('test4')
            cohort_temp_class = []
#             print(range(len(tracks)))
            for j in range(len(tracks)):
#                 print(len(class_tmp))
#                 if j in class_tmp:
# #                     print('test')
                if j in class_tmp and return_track_attributes.return_track_lifetime(tracks,j) >= cohort[0] and return_track_lifetime(tracks,j) < cohort[1]: # find tracks within bounds
#                     print('test5')
                    ch0 = return_track_amplitude_one_channel(tracks, j, 0)
                    ch1 = return_track_amplitude_one_channel(tracks, j, 1)
                    
                    if alignment=='interpolate':
#                         print('test3')
                        t0, c0, k0 = interpolate.splrep(np.arange(len(ch0)), ch0, k=3) # cubic B-spline iterpolation of intensities
                        t1, c1, k1 = interpolate.splrep(np.arange(len(ch1)), ch1, k=3)

                        spline0 = interpolate.BSpline(t0, c0, k0, extrapolate=False)
                        spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)

                        splined_amps_ch0 = spline0(np.linspace(0,len(ch0),cohort[1])) # interpolate intensity to match ceiling of lifetime cohort
                        splined_amps_ch1 = spline1(np.linspace(0,len(ch1),cohort[1]))

                        cohort_temp_class.append([splined_amps_ch0, splined_amps_ch1])
                        
                    elif alignment=='maxCh1':
                        
                        cohort_temp_class.append([ch0,ch1])
                        
            if alignment=='maxCh1':
                
                frames_before_alignment = []
                frames_after_alignment = []
                
                for track_index in range(len(cohort_temp_class)):
                    
                    frames_before_alignment.append(np.argmax(cohort_temp_class[track_index][1]))
                    frames_after_alignment.append(len(cohort_temp_class[track_index][1]) - np.argmax(cohort_temp_class[track_index][1]) - 1)

                padding_amount = np.max([np.max(frames_before_alignment), np.max(frames_after_alignment)])
#                 padding_amount = np.max([padding_amount,])
                cohort_temp_class = alignment_tools.return_shifted_amplitudes(cohort_temp_class, 
                                                                         frames_before_alignment, 
                                                                         frames_after_alignment, 
                                                                         padding_amount, 
                                                                         True)


            if alignment=='maxCh1':
            
                                        
                if cohort_temp_class != []:
            
    #                 print(len(cohort_temp_class))
                    average_cohort_class = np.nan_to_num(np.nanmean(cohort_temp_class,axis=0,dtype=np.float64)) # calculate average and std of intensity in class cohort
                    std_cohort_class = num_stds*np.nan_to_num(np.nanstd(cohort_temp_class,axis=0,dtype=np.float64))

                    ch0 = average_cohort_class[0,:]
                    ch1 = average_cohort_class[1,:]

                    t0, c0, k0 = interpolate.splrep(np.arange(len(ch0)), ch0, k=3) # cubic B-spline iterpolation of intensities
                    t1, c1, k1 = interpolate.splrep(np.arange(len(ch1)), ch1, k=3)

                    spline0 = interpolate.BSpline(t0, c0, k0, extrapolate=False)
                    spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)

                    splined_amps_ch0 = spline0(np.linspace(0,len(ch0),cohort[1])) # interpolate intensity to match ceiling of lifetime cohort
                    splined_amps_ch1 = spline1(np.linspace(0,len(ch1),cohort[1]))

                    avg_intensities_ch0 = splined_amps_ch0.copy()
                    avg_intensities_ch1 = splined_amps_ch1.copy()

                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                                splined_amps_ch0,
                                'm', 
                                label=str('Cohort: ' + str(cohort) + ' s members: ' + '{:n}'.format(str(len(cohort_temp_class)))))

                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                            splined_amps_ch1,
                            'g')

                    ch0 = std_cohort_class[0,:]
                    ch1 = std_cohort_class[1,:]

                    t0, c0, k0 = interpolate.splrep(np.arange(len(ch0)), ch0, k=3) # cubic B-spline iterpolation of intensities
                    t1, c1, k1 = interpolate.splrep(np.arange(len(ch1)), ch1, k=3)

                    spline0 = interpolate.BSpline(t0, c0, k0, extrapolate=False)
                    spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)

                    splined_amps_ch0 = spline0(np.linspace(0,len(ch0),cohort[1])) # interpolate intensity to match ceiling of lifetime cohort
                    splined_amps_ch1 = spline1(np.linspace(0,len(ch1),cohort[1]))


                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    avg_intensities_ch0-splined_amps_ch0, 
                                    avg_intensities_ch0+splined_amps_ch0, 
                                    color='m', 
                                    alpha=0.2)
                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    avg_intensities_ch1-splined_amps_ch1, 
                                    avg_intensities_ch1+splined_amps_ch1, 
                                    color='g', 
                                    alpha=0.2)

                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('fluorescence intensity')
                    ax.legend()
#                 cohort_temp_class.append([splined_amps_ch0, splined_amps_ch1])
                
#                 print(len(average_cohort_class[0]))
#                 x_axis = np.linspace(cohort[1]-len(average_cohort_class[0]),cohort[1],len(average_cohort_class[0]))
#                 print(len(x_axis))
#                 print(len(average_cohort_class[0,:]))
#                 ax.plot(x_axis, 
#                         average_cohort_class[0,:],
#                         'm', 
#                         label=str('Cohort: ' + str(cohort) + ' s members: ' + str(len(cohort_temp_class))))
#                 ax.plot(x_axis, 
#                         average_cohort_class[1,:],
#                         'g')
#                 ax.fill_between(x_axis, 
#                                 average_cohort_class[0,:]-std_cohort_class[0,:], 
#                                 average_cohort_class[0,:]+std_cohort_class[0,:], 
#                                 color='m', 
#                                 alpha=0.2)
#                 ax.fill_between(x_axis, 
#                                 average_cohort_class[1,:]-std_cohort_class[1,:], 
#                                 average_cohort_class[1,:]+std_cohort_class[1,:], 
#                                 color='g', 
#                                 alpha=0.2)
#                 ax.title.set_text('number of members in class ' + str(i) + ': ' + str(len(class_tmp)))
#                 ax.set_xlabel('time (s)')
#                 ax.set_ylabel('fluorescence intensity')
#                 ax.legend()        
    
            elif alignment=='interpolate':   
#                 print('test')
                if cohort_temp_class != []:
#                     print('test2')
                    print()       
                    average_cohort_class = np.nan_to_num(np.nanmean(cohort_temp_class,axis=0,dtype=np.float64)) # calculate average and std of intensity in class cohort
                    std_cohort_class = num_stds*np.nan_to_num(np.nanstd(cohort_temp_class,axis=0,dtype=np.float64))
                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                            average_cohort_class[0,:],
                            'm', 
                            label=str('Cohort: ' + str(cohort) + ' s members: ' + str(len(cohort_temp_class))))
                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                            average_cohort_class[1,:],
                            'g')
                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    average_cohort_class[0,:]-std_cohort_class[0,:], 
                                    average_cohort_class[0,:]+std_cohort_class[0,:], 
                                    color='m', 
                                    alpha=0.2)
                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    average_cohort_class[1,:]-std_cohort_class[1,:], 
                                    average_cohort_class[1,:]+std_cohort_class[1,:], 
                                    color='g', 
                                    alpha=0.2)
                    ax.title.set_text('number of members in class ' + str(i) + ': ' + str(len(class_tmp)))
                    ax.set_xlabel('time (s)')
                    ax.set_ylabel('fluorescence intensity')
                    ax.legend()
                
            else:
                    
                raise Exception('alignment option does not exist')
        
        subplot_index += 1
                
                        
    if filename != '': # save the plot
        
        f.savefig(filename)
        
    plt.show()

def fit_cohorts_fixed_axes(tracks, 
                class_indices, 
                upper_bounds,
                figsize, 
                plot_shape, 
                num_stds = 0.25, 
                cohorts=[[0,20],[21,40],[41,60],[61,80],[81,100],[101,120],[121,140],[141,160],[161,180],[181,200],[201,220],[221,240]], 
                filename = '',
                alignment='interpolate'):
    """
    Display lifetime-binned cohorts of tracks separated by different class labels. 
    Tracks in each cohort are interpolated to the length of the ceiling of the cohort.
    
    Args:
        tracks (ndarray): numpy array of cmeAnalysis' ProcessedTracks.mat 
        class_indices (list of lists): each internal list contains indices for tracks in a designated class
        upper_bounds (float): upper bounds of a.u. fluorescence (y) axis
        figsize (tuple): figure size for all subplots
        plot_shape (tuple): size of each subplot
        num_stds (float, optional): number of standard deviations of intensity for plotting intensity traces
        cohorts (list of lists, optional): each internal list contains the upper and lower bounds of lifetimes for each cohort
        filename (string, optional): location for plots to be saved
        interpolate (string, optional): method for averaging intensities in cohorts
        
    Returns:
        None
        
    """
    plt.rcParams.update({'font.size': 30})

    print('fitting to the following cohorts: ' + str(cohorts))
    print()
    f = plt.figure(dpi=300,figsize=figsize)
    subplot_index = 1
    warnings.filterwarnings("ignore")

    class_intensities = []
    
    for i,class_tmp in enumerate(class_indices): # iterate through each class of labels
    
        class_intensities_current = []
        
        ax = f.add_subplot(plot_shape[0], plot_shape[1], subplot_index)
        ax.set_title('class: ' + str(i))
        
        for cohort in cohorts: # gather intensities for each cohort for all members of current class
#             print('test4')
            cohort_temp_class = []
#             print(range(len(tracks)))
            for j in range(len(tracks)):
#                 print(len(class_tmp))
#                 if return_track_attributes.return_track_lifetime(tracks,j) >= cohort[0]:
#                     print('test')
#                 print(cohort[0])
#                 print(return_track_attributes.return_track_lifetime(tracks,j))
                if j in class_tmp and return_track_attributes.return_track_lifetime(tracks,j) >= cohort[0] and return_track_lifetime(tracks,j) < cohort[1]: # find tracks within bounds
#                     print('test5')
                    ch0 = return_track_amplitude_one_channel(tracks, j, 0)
                    ch1 = return_track_amplitude_one_channel(tracks, j, 1)
                    
                    if alignment=='interpolate':
#                         print('test3')
                        t0, c0, k0 = interpolate.splrep(np.arange(len(ch0)), ch0, k=3) # cubic B-spline iterpolation of intensities
                        t1, c1, k1 = interpolate.splrep(np.arange(len(ch1)), ch1, k=3)

                        spline0 = interpolate.BSpline(t0, c0, k0, extrapolate=False)
                        spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)

                        splined_amps_ch0 = spline0(np.linspace(0,len(ch0),cohort[1])) # interpolate intensity to match ceiling of lifetime cohort
                        splined_amps_ch1 = spline1(np.linspace(0,len(ch1),cohort[1]))

                        cohort_temp_class.append([splined_amps_ch0, splined_amps_ch1])
                        
                    elif alignment=='maxCh1':
                        
                        cohort_temp_class.append([ch0,ch1])
                        
            if alignment=='maxCh1':
                
                frames_before_alignment = []
                frames_after_alignment = []
                
                for track_index in range(len(cohort_temp_class)):
                    
                    frames_before_alignment.append(np.argmax(cohort_temp_class[track_index][1]))
                    frames_after_alignment.append(len(cohort_temp_class[track_index][1]) - np.argmax(cohort_temp_class[track_index][1]) - 1)

                padding_amount = np.max([np.max(frames_before_alignment), np.max(frames_after_alignment)])
#                 padding_amount = np.max([padding_amount,])
                cohort_temp_class = alignment_tools.return_shifted_amplitudes(cohort_temp_class, 
                                                                         frames_before_alignment, 
                                                                         frames_after_alignment, 
                                                                         padding_amount, 
                                                                         True)


            if alignment=='maxCh1':
            
                                        
                if cohort_temp_class != []:
            
    #                 print(len(cohort_temp_class))
                    average_cohort_class = np.nan_to_num(np.nanmean(cohort_temp_class,axis=0,dtype=np.float64)) # calculate average and std of intensity in class cohort
                    std_cohort_class = num_stds*np.nan_to_num(np.nanstd(cohort_temp_class,axis=0,dtype=np.float64))

                    ch0 = average_cohort_class[0,:]
                    ch1 = average_cohort_class[1,:]

                    t0, c0, k0 = interpolate.splrep(np.arange(len(ch0)), ch0, k=3) # cubic B-spline iterpolation of intensities
                    t1, c1, k1 = interpolate.splrep(np.arange(len(ch1)), ch1, k=3)

                    spline0 = interpolate.BSpline(t0, c0, k0, extrapolate=False)
                    spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)

                    splined_amps_ch0 = spline0(np.linspace(0,len(ch0),cohort[1])) # interpolate intensity to match ceiling of lifetime cohort
                    splined_amps_ch1 = spline1(np.linspace(0,len(ch1),cohort[1]))

                    avg_intensities_ch0 = splined_amps_ch0.copy()
                    avg_intensities_ch1 = splined_amps_ch1.copy()

                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                                splined_amps_ch0,
                                'm', 
                                label=str('Cohort: ' + str(cohort) + ' s members: ' + '{:,}'.format(len(cohort_temp_class))))

                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                            splined_amps_ch1,
                            'g')

                    ch0 = std_cohort_class[0,:]
                    ch1 = std_cohort_class[1,:]

                    t0, c0, k0 = interpolate.splrep(np.arange(len(ch0)), ch0, k=3) # cubic B-spline iterpolation of intensities
                    t1, c1, k1 = interpolate.splrep(np.arange(len(ch1)), ch1, k=3)

                    spline0 = interpolate.BSpline(t0, c0, k0, extrapolate=False)
                    spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)

                    splined_amps_ch0 = spline0(np.linspace(0,len(ch0),cohort[1])) # interpolate intensity to match ceiling of lifetime cohort
                    splined_amps_ch1 = spline1(np.linspace(0,len(ch1),cohort[1]))


                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    avg_intensities_ch0-splined_amps_ch0, 
                                    avg_intensities_ch0+splined_amps_ch0, 
                                    color='m', 
                                    alpha=0.2)
                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    avg_intensities_ch1-splined_amps_ch1, 
                                    avg_intensities_ch1+splined_amps_ch1, 
                                    color='g', 
                                    alpha=0.2)

                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Fluorescence (a.u.)')
                    ax.legend()
                    
                    plt.tight_layout()
      
    
            elif alignment=='interpolate':   
#                 print('test')
                if cohort_temp_class != []:
#                     print('test2')
                    print()       
                    average_cohort_class = np.nan_to_num(np.nanmean(cohort_temp_class,axis=0,dtype=np.float64)) # calculate average and std of intensity in class cohort
                    std_cohort_class = num_stds*np.nan_to_num(np.nanstd(cohort_temp_class,axis=0,dtype=np.float64))
                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                            average_cohort_class[0,:],
                            'm', 
                            label=str('Cohort: ' + str(cohort) + ' s members: ' + '{:,}'.format(len(cohort_temp_class))))
                    ax.plot(np.linspace(0,cohort[1],cohort[1]), 
                            average_cohort_class[1,:],
                            'g')
                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    average_cohort_class[0,:]-std_cohort_class[0,:], 
                                    average_cohort_class[0,:]+std_cohort_class[0,:], 
                                    color='m', 
                                    alpha=0.2)
                    ax.fill_between(np.linspace(0,cohort[1],cohort[1]), 
                                    average_cohort_class[1,:]-std_cohort_class[1,:], 
                                    average_cohort_class[1,:]+std_cohort_class[1,:], 
                                    color='g', 
                                    alpha=0.2)
                    ax.title.set_text('number of members in cluster ' + str(i) + ': ' + '{:,}'.format(len(class_tmp)))
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Fluorescence (a.u.)')
                    ax.legend()
                    ax.set_ylim([0,upper_bounds])
                    
                    plt.tight_layout()
                
            else:
                    
                raise Exception('alignment option does not exist')
        
        subplot_index += 1
                
                        
    if filename != '': # save the plot
        
        f.savefig(filename)
        
    plt.show()
    
def load_tracks(track_path):
    """Return a Python structure of the tracks, sorted from decreasing lifetime, from the ProcessedTracks.mat output of cmeAnalysis, with keys designated track indices"""
    return sort_tracks_descending_lifetimes(sio.loadmat(track_path)) # sort the tracks by descending lifetime order

def load_tracks_no_sort(track_path):
    """Return a dictionary of the tracks from the ProcessedTracks.mat output of cmeAnalysis, with keys designated track indices"""
    return sio.loadmat(track_path)['tracks'][0] # convert the MATLAB structure to a Python one

def sort_tracks_descending_lifetimes(tracks):
    """Sort tracks in descending lifetime order"""
    index_dictionary = return_index_dictionary()
    
    tracks = tracks['tracks'][0] # get just the data for the tracks
                                            
    tracks = zip(tracks, range(len(tracks))) # save each track with its original index
    # sort the tracks in descending lifetime, preserving each track's individual index
    tracks = sorted(tracks, key=lambda track: track[0][index_dictionary['index_lifetime_s']][0][0], reverse=True) 
    
    (tracks, indices) = zip(*tracks) 
    
    return tracks

def return_check_box_value(make_gif,
                           lower_frame_bound,
                           upper_frame_bound,
                           window_size,
                           tracks,
                           raw_tiff_list,
                           track_number,
                           channel_colors,
                           display_channels,
                           frame_names,
                           gif_name,
                           folder_name,
                           dpi,
                           subplot_padding):
    """Make a gif of the current visualized state"""
    print('Enter the upper and lower bounds of frames for the generated gif')
    print('The folder name and gif name by default will be appended by "track_number_frame_START_to_END"')
    if make_gif: # if the gif making checkbox is selected, make a gif
        print('creating gif')
        play_through_tracks(tracks,
                            raw_tiff_list,
                            track_number,
                            channel_colors,
                            lower_frame_bound,
                            upper_frame_bound,
                            window_size,
                            display_channels,
                            frame_names,
                            gif_name,
                            folder_name,
                            dpi,
                            subplot_padding)


def load_itk_image(image_path):
    """Load a fluorescent multiframe image as an array"""
    image = itk.imread(image_path) # read an image

    image = itk.array_view_from_image(image) # and convert it into an array
    
    return image

def collect_plots_to_make_per_track(display_intensity,
                                    display_movement,
                                    display_distance_traveled,
                                    display_separation):
    """Designate the plot(s) to make from user selection(s)"""
    display_attributes = []

    if display_intensity:
        
        display_attributes.append('display_intensity')
    
    if display_movement:
        
        display_attributes.append('display_movement')
        
    if display_distance_traveled:
        
        display_attributes.append('display_distance_traveled')
        
    if display_separation:
        
        display_attributes.append('display_separation')

    return display_attributes

def display_intensity_subplot(ax_current,
                              tracks,
                              track_number,
                              channel_colors,
                              display_channels):
    """Plot intensity over time for all channels in a track"""
    for channel_number in display_channels:

        ax_current.plot(return_track_amplitude_one_channel(tracks,
                                                    track_number,
                                                    channel_number),
                        color=channel_colors[channel_number],
                        label='ch' + str(channel_number))

    ax_current.set(xlabel='frames', ylabel='au fluorescence intensity')
    ax_current.legend()
    
    

def display_movement_subplot(ax_current,
                              tracks,
                              track_number,
                              channel_colors,
                              display_channnels):
    """Plot x(t), y(t) in pixels for all designated channels in a track"""            
    marker_channel = ['v','^','<','>']
    for channel_number in display_channnels:
        # plot whole channel's trace
        ax_current.plot(return_puncta_x_position_whole_track(tracks,
                                                      track_number,
                                                      channel_number),
                        return_puncta_y_position_whole_track(tracks,
                                                      track_number,
                                                      channel_number),
                        color=channel_colors[channel_number],
                        label='ch' + str(channel_number),
                        alpha = 0.5)  
        # signify the beginning of channel's track
        ax_current.plot(return_puncta_x_position_whole_track(tracks,
                                                      track_number,
                                                      channel_number)[0],
                        return_puncta_y_position_whole_track(tracks,
                                                      track_number,
                                                      channel_number)[0],
                        color=channel_colors[channel_number],
                        label='ch' + str(channel_number) + ' start',
                        marker=marker_channel[channel_number],
                        markerSize = 10,
                        alpha = 0.7)
        # signify the end of a channel's track
        ax_current.plot(return_puncta_x_position_whole_track(tracks,
                                                      track_number,
                                                      channel_number)[-1],
                        return_puncta_y_position_whole_track(tracks,
                                                      track_number,
                                                      channel_number)[-1],
                        color=channel_colors[channel_number],
                        label='ch' + str(channel_number) + ' end',
                        marker=marker_channel[channel_number],
                        markerSize = 20,
                        alpha = 0.7)


    ax_current.set(xlabel='x position (pixel)', ylabel='y position (pixel)')
    ax_current.legend()

    
def distance_travelled_subplot(ax_current,
                               tracks,
                               track_number,
                               channel_colors,
                               display_channels):
    
    """Plot distance travelled (in pixels) from origin of track for all channels in a track"""   
    for channel_number in display_channels:
        
        ax_current.plot(return_distance_traveled_from_origin(tracks,
                                                              track_number,
                                                              channel_number),
                                                              color=channel_colors[channel_number],
                                                              label='ch' + str(channel_number))   
        
    ax_current.set(xlabel='frames',ylabel='distance from channel origin (pixels)')
    ax_current.legend()
    
def distance_separated_between_channels_subplot(ax_current,
                                                tracks,
                                                track_number,
                                                channel_colors,
                                                first_separation_channel,
                                                second_separation_channel,
                                                display_channels):
    """Plot distance between (in pixels) the fitted centers of two designated channels in a track"""  
    for channel_number in display_channels:
                
        ax_current.plot(return_distance_between_two_channel(tracks,
                                                            track_number,
                                                            first_separation_channel,
                                                            second_separation_channel),
                                                            color=channel_colors[channel_number])
    
    ax_current.set(xlabel='frames',ylabel='distance between designated channels (pixels)')    
    
    
def display_one_track(tracks, 
                      track_number, 
                      dpi, 
                      fig_size, 
                      channel_colors, 
                      display_channels,
                      display_intensity,
                      display_movement,
                      display_distance_traveled,
                      display_separation,
                      show_frames,
                      create_gif,
                      raw_frames,
                      first_separation_channel,
                      second_separation_channel):
    """
    Visualize a single track's designated features by user input
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        track_number (int): index of track for display
        dpi (int): dpi for matplotlib figure
        fig_size (tuple): size of matplotlib figure
        channel_colors (list): colors for plotting traces corresponding to each displayed channel
        display_channels (list): channels to display from tracks
        display_intensity (boolean): display intensity vs. time 
        display_movement (boolean): display (x(t),y(t)) in pixels
        display_distance_traveled (boolean): display distance traveled from origin of track
        display_separation (boolean): display separation between two channels' fitted center vs. time 
        show_frames (boolean): show raw data centered around track
        create_gif (boolean): create gif of raw data separated by channels and intensity vs. time for designated chnanels 
        raw_frames (list of strings): list of strings of pathnames for raw data channels (must contain all channels for tracks)
        first_separation_channel (int): index of first channel for separation display
        second_separation_channel (int): index of second chnanel for separation display
        
    Returns:
        None
    
    """
    fig=figure(num=None, figsize=fig_size, dpi=dpi, facecolor='w', edgecolor='k')

    num_channels_in_track = len(return_track_amplitude(tracks,track_number))

    print('The TrackID of the following track is: ' + str(track_number))

    
    display_attributes = collect_plots_to_make_per_track(display_intensity,
                                                         display_movement,
                                                         display_distance_traveled,
                                                         display_separation)

    marker_channel = ['v','^','<','>']
    
    index_subplot = 1 # the index of the first subplot of plot size (1*n) where n = len(display_attributes)
    number_channels_in_track = len(return_track_amplitude(tracks, track_number))
    
    for i in range(len(display_attributes)):
        
        ax_current = fig.add_subplot(1,len(display_attributes),index_subplot)
        
        if display_attributes[i]=='display_intensity':
            
            display_intensity_subplot(ax_current,
                              tracks,
                              track_number,
                              channel_colors,
                              display_channels)
            
        elif display_attributes[i]=='display_movement':    
            
            display_movement_subplot(ax_current,
                                  tracks,
                                  track_number,
                                  channel_colors,
                                  display_channels)    

        elif display_attributes[i]=='display_distance_traveled':
            
            distance_travelled_subplot(ax_current,
                                       tracks,
                                       track_number,
                                       channel_colors,
                                       display_channels)

        elif display_attributes[i]=='display_separation':   
            
            distance_separated_between_channels_subplot(ax_current,
                                                    tracks,
                                                    track_number,
                                                    channel_colors,
                                                    first_separation_channel,
                                                    second_separation_channel,
                                                    display_channels)
        index_subplot+=1
    
    track_category=return_track_category(tracks, track_number)
    track_is_ccp=return_is_CCP(tracks, track_number)
    print(f'The category of this track track is: {track_category}')   
    if track_is_ccp:
        print('This track was categorized as a CCP by cmeAnalysis')
    else:
        print('This track was not categorized as a CCP by cmeAnalysis')
    plt.show()  
    
    if show_frames: # if user selects to see raw frames overlaid with puncta fits

        interact_through_tracks_IDs(tracks,
                                    raw_frames,
                                    track_number,
                                    channel_colors,
                                    create_gif,
                                    display_channels)
    


        
def play_through_tracks(tracks,raw_images,track_number,channel_colors,l,u,window_size,display_channels, frame_names, gif_name,folder_name,dpi, subplot_padding):
    """Create a folder for raw data gif frames and construct gif"""
    
    fig=plt.figure(num=None, figsize=(50,10), dpi=40, facecolor='w', edgecolor='k')   
    folder_name = folder_name+'_track_number_'+str(track_number)+'_frame_'+str(l)+'_to_'+str(u)
    list_of_frames = []

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        raise Exception('Folder exists in current directory')

    for frame in range(l,u+1):
        
        interact_through_frames_of_track_save_figure(tracks,
                                                     raw_images,
                                                     track_number,
                                                     frame,
                                                     window_size,
                                                     channel_colors,
                                                     dpi,
                                                     display_channels,
                                                     folder_name,
                                                     frame_names,
                                                     subplot_padding)

    print(folder_name)

    image_list = []
    
    for i in range(l,u+1):
        
        image_list.append(imageio.imread(folder_name+'/'+frame_names+str(i)+'.png'))

    print('completed frame generation; building gif')
    imageio.mimwrite(folder_name+'/'+gif_name + '_track_number_' + str(track_number)+'_frame_'+str(l)+'_to_'+str(u)+'.gif', image_list,fps=3)
    print('gif generation complete')

        
def interact_through_tracks_IDs(tracks,raw_images,track_number,channel_colors, create_gif, display_channels):
    """Set up widgets to interact through a track's frames and display raw data"""
    len_current_track = len(return_track_amplitude_one_channel(tracks,track_number,0))
    print('The length of the current track: {}'.format(len_current_track))

    interact(interact_through_frames_of_track,
             raw_images=fixed(raw_images),
             track_number=fixed(track_number),
             tracks=fixed(tracks),
             frames=widgets.IntSlider(min=0,max=len_current_track-1,step=1,value=0,continuous_update=False),
             window_size=widgets.IntSlider(min=-10,max=30,step=1,value=10),
             subplot_padding=widgets.IntSlider(min=20,max=100,step=1,value=25),
             channel_colors=fixed(channel_colors),
             dpi=widgets.IntSlider(value=50,min=0,max=1000,step=1,description='dpi of raw images:',disabled=False),
             create_gif=fixed(create_gif),
             display_channels=fixed(display_channels))

# 
def interact_through_frames_of_track_save_figure(tracks,
                                                 raw_images, 
                                                 track_number,
                                                 frames,
                                                 window_size, 
                                                 channel_colors, 
                                                 dpi, 
                                                 display_channels,
                                                 folder_name,
                                                 frame_names,
                                                 subplot_padding):
    """Display raw frames interactively and prompt user for gif-saving options"""
    print('Current frame: ' + str(frames))
    warnings.filterwarnings("ignore")
    num_channels = len(return_track_amplitude(tracks,track_number))
    fig, axes = plt.subplots(1, len(display_channels)+2, figsize=(40,5),gridspec_kw = {'wspace':0.1, 'hspace':0.1}, dpi=dpi)
    index_subplot = 0
    colors=['Reds','Greens','Blues']
    color_puncta = channel_colors
    track_x_positions = []
    track_y_positions = []
    frames_in_track = return_frames_in_track(tracks,track_number)-1
    min_x = np.min(return_puncta_x_position_whole_track(tracks,track_number,0))
    max_x = np.max(return_puncta_x_position_whole_track(tracks,track_number,0))
    min_y = np.min(return_puncta_y_position_whole_track(tracks,track_number,0))
    max_y = np.max(return_puncta_y_position_whole_track(tracks,track_number,0))
    for i in range(num_channels):

        track_x_positions.append(return_puncta_x_position_whole_track(tracks,track_number,i)-0.5)
        track_y_positions.append(return_puncta_y_position_whole_track(tracks,track_number,i)-0.5)
        if np.min(return_puncta_x_position_whole_track(tracks,track_number,i))<min_x:
            min_x = np.min(return_puncta_x_position_whole_track(tracks,track_number,i))
        if np.max(return_puncta_x_position_whole_track(tracks,track_number,i))>max_x:
            max_x = np.max(return_puncta_x_position_whole_track(tracks,track_number,i))
        if np.min(return_puncta_y_position_whole_track(tracks,track_number,i))<min_y:
            min_y = np.min(return_puncta_y_position_whole_track(tracks,track_number,i))
        if np.max(return_puncta_y_position_whole_track(tracks,track_number,i))>max_y:
            max_y = np.max(return_puncta_y_position_whole_track(tracks,track_number,i))
    diff_x = max_x-min_x
    diff_y = max_y-min_y
    avg_x = (max_x+min_x)/2
    avg_y = (max_y+min_y)/2
    diff_greatest = int(np.max([diff_x,diff_y])/2+window_size)

    lower_x_lim = np.max([0,int(avg_x-diff_greatest)])
    lower_y_lim = np.max([0,int(avg_y-diff_greatest)])
    upper_x_lim = np.min([511,int(avg_x+diff_greatest)])
    upper_y_lim = np.min([511,int(avg_y+diff_greatest)])
    
    
    for i in display_channels:

        frame = load_itk_image(raw_images[i])[frames_in_track[frames],:,:][lower_y_lim:upper_y_lim,lower_x_lim:upper_x_lim]
        
        axes[index_subplot].imshow(frame,cmap='Greys')
        axes[index_subplot].title.set_text('Ch'+str(i))
        axes[index_subplot].set_xlabel('pixels')
        axes[index_subplot].set_ylabel('pixels')
        for j in display_channels:

            axes[index_subplot].plot(track_x_positions[j][frames]-lower_x_lim,track_y_positions[j][frames]-lower_y_lim,marker='X',markerSize=2,color=color_puncta[j],label='Ch'+str(j))
        
        axes[index_subplot].legend()
        index_subplot+=1 
        

    
    for i in display_channels:
        
        frame = load_itk_image(raw_images[i])[frames_in_track[frames],:,:][lower_y_lim:upper_y_lim,lower_x_lim:upper_x_lim]

        axes[index_subplot].imshow(frame,cmap=colors[i],alpha=0.6)
        axes[index_subplot].plot(track_x_positions[i][frames]-lower_x_lim,track_y_positions[i][frames]-lower_y_lim,marker='x',markerSize=2,label='Ch'+str(j),color=color_puncta[i])      


    display_overlay_string='overlay, '
    for i in range(num_channels):
        if i in display_channels:
            display_overlay_string+='Ch'+str(i)+':'+str(colors[i])+' '
    axes[index_subplot].title.set_text(display_overlay_string)
    axes[index_subplot].set_xlabel('pixels')
    axes[index_subplot].set_ylabel('pixels')
    axes[index_subplot].legend()

    index_subplot += 1

    for i in display_channels:
        
        axes[index_subplot].plot(return_track_amplitude_one_channel(tracks,track_number,i),color=channel_colors[i],label='Ch'+str(i))

    axes[index_subplot].axvline(frames,label='current frame')
    axes[index_subplot].legend(loc=1)
    axes[index_subplot].set_xlabel('frames')
    axes[index_subplot].set_ylabel('au fluorescence intensity')
    axes[index_subplot].title.set_text('fluorescence intensity over time frame = ' + str(frames))

    fig.tight_layout()
    plt.savefig(folder_name + '/'+ frame_names +str(frames))
    

def interact_through_frames_of_track(tracks,
                                     raw_images, 
                                     track_number,
                                     frames,
                                     window_size, 
                                     channel_colors, 
                                     dpi, 
                                     create_gif, 
                                     display_channels,
                                     subplot_padding):
    """Interact through a track's frames and display raw data"""
    print('Current frame: ' + str(frames))
    warnings.filterwarnings("ignore")
    num_channels = len(return_track_amplitude(tracks,track_number))
    fig, axes = plt.subplots(1, len(display_channels)+2, figsize=(subplot_padding,5),gridspec_kw = {'wspace':0.1, 'hspace':0.1}, dpi=dpi)
    index_subplot = 0
    colors=['Reds','Greens','Blues']
    color_puncta = channel_colors
    track_x_positions = []
    track_y_positions = []
    frames_in_track = return_frames_in_track(tracks,track_number)-1
    min_x = np.min(return_puncta_x_position_whole_track(tracks,track_number,0))
    max_x = np.max(return_puncta_x_position_whole_track(tracks,track_number,0))
    min_y = np.min(return_puncta_y_position_whole_track(tracks,track_number,0))
    max_y = np.max(return_puncta_y_position_whole_track(tracks,track_number,0))
    for i in range(num_channels):

        track_x_positions.append(return_puncta_x_position_whole_track(tracks,track_number,i)-0.5)
        track_y_positions.append(return_puncta_y_position_whole_track(tracks,track_number,i)-0.5)
        if np.min(return_puncta_x_position_whole_track(tracks,track_number,i))<min_x:
            min_x = np.min(return_puncta_x_position_whole_track(tracks,track_number,i))
        if np.max(return_puncta_x_position_whole_track(tracks,track_number,i))>max_x:
            max_x = np.max(return_puncta_x_position_whole_track(tracks,track_number,i))
        if np.min(return_puncta_y_position_whole_track(tracks,track_number,i))<min_y:
            min_y = np.min(return_puncta_y_position_whole_track(tracks,track_number,i))
        if np.max(return_puncta_y_position_whole_track(tracks,track_number,i))>max_y:
            max_y = np.max(return_puncta_y_position_whole_track(tracks,track_number,i))
    diff_x = max_x-min_x
    diff_y = max_y-min_y
    avg_x = (max_x+min_x)/2
    avg_y = (max_y+min_y)/2
    diff_greatest = int(np.max([diff_x,diff_y])/2+window_size)

    lower_x_lim = np.max([0,int(avg_x-diff_greatest)])
    lower_y_lim = np.max([0,int(avg_y-diff_greatest)])
    upper_x_lim = np.min([511,int(avg_x+diff_greatest)])
    upper_y_lim = np.min([511,int(avg_y+diff_greatest)])
    
    
    for i in display_channels:

        frame = load_itk_image(raw_images[i])[frames_in_track[frames],:,:][lower_y_lim:upper_y_lim,lower_x_lim:upper_x_lim]

        axes[index_subplot].imshow(frame,cmap='Greys')
        axes[index_subplot].title.set_text('Ch'+str(i))
        axes[index_subplot].set_xlabel('pixels')
        axes[index_subplot].set_ylabel('pixels')
        for j in display_channels:
            axes[index_subplot].plot(track_x_positions[j][frames]-lower_x_lim,track_y_positions[j][frames]-lower_y_lim,marker='X',markerSize=2,color=color_puncta[j],label='Ch'+str(j))
        
        axes[index_subplot].legend()
        index_subplot+=1 
    
    for i in display_channels:
        frame = load_itk_image(raw_images[i])[frames_in_track[frames],:,:][lower_y_lim:upper_y_lim,lower_x_lim:upper_x_lim]

        axes[index_subplot].imshow(frame,cmap=colors[i],alpha=0.6)
        axes[index_subplot].plot(track_x_positions[i][frames]-lower_x_lim,track_y_positions[i][frames]-lower_y_lim,marker='x',markerSize=2,label='Ch'+str(j),color=color_puncta[i])      

    display_overlay_string='overlay, '
    for i in range(num_channels):
        if i in display_channels:
            display_overlay_string+='Ch'+str(i)+':'+str(colors[i])+' '
    axes[index_subplot].title.set_text(display_overlay_string)
    axes[index_subplot].set_xlabel('pixels')
    axes[index_subplot].set_ylabel('pixels')
    axes[index_subplot].legend()
    index_subplot += 1
    for i in display_channels:
        
        axes[index_subplot].plot(return_track_amplitude_one_channel(tracks,track_number,i),color=channel_colors[i],label='Ch'+str(i))

    axes[index_subplot].axvline(frames,label='current frame')
    axes[index_subplot].legend(loc=1)
    axes[index_subplot].set_xlabel('frames')
    axes[index_subplot].set_ylabel('au fluorescence intensity')
    axes[index_subplot].title.set_text('fluorescence intensity over time, frame = ' + str(frames))

    fig.tight_layout()
    fig.show()        
    
    if create_gif: # if user selects to turn frames into a saved gif
        len_current_track = len(return_track_amplitude_one_channel(tracks,track_number,0))
        w = widgets.Checkbox(value=False, description='Click to create GIF')
        lower_bound=widgets.BoundedIntText(value=0,
                                           min=0,
                                           max=len(return_track_amplitude_one_channel(tracks,track_number,0))-2,
                                           step=1,
                                           description='lower:',
                                           style={'width': 'max-content'},
                                           disabled=False)

        upper_bound=widgets.BoundedIntText(value=len(return_track_amplitude_one_channel(tracks,track_number,0)),
                                           min=1,
                                           max=len(return_track_amplitude_one_channel(tracks,track_number,0))-1,
                                           step=1,
                                           description='upper:',
                                           style={'width': 'max-content'}, 
                                           disabled=False)


        frame_names=widgets.Textarea(value='frame_',
                             placeholder='frame_',
                             description='frame name prefix:',
                             disabled=False)
        gif_name=widgets.Textarea(value='gif',
                             placeholder='gif',
                             description='gif name:',
                             disabled=False)

        folder_name=widgets.Textarea(value='saved_pngs',
                                  placeholder='saved_pngs',
                                  description='Folder name:',
                                  disabled=False)
        interact(return_check_box_value,
                 make_gif=w,
                 lower_frame_bound=lower_bound,
                 upper_frame_bound=upper_bound,
                 frame_names=frame_names,
                 gif_name=gif_name,
                 folder_name=folder_name,
                 window_size=fixed(window_size),
                 tracks=fixed(tracks),
                 raw_tiff_list=fixed(raw_images),
                 track_number=fixed(track_number),
                 channel_colors=fixed(channel_colors),
                 display_channels=fixed(display_channels),
                 dpi=fixed(dpi),
                 subplot_padding=fixed(subplot_padding))   

def track_within_bounds(tracks, track_number, minimum_lifetime, maximum_lifetime, track_category):
    """Check if a track satisfies the user-defined conditions"""
    if return_track_category(tracks, track_number) in track_category and \
       return_track_lifetime(tracks, track_number) >= minimum_lifetime and \
       return_track_lifetime(tracks, track_number) <= maximum_lifetime:

        return True

    else:

        return False

    
def display_tracks_for_selection(tracks):
    """Display shuffled tracks for manual user selection of tracks indices to keep for further analysis"""
    indices = list(range(len(tracks)))
    random.shuffle(indices)
    print('Tracks have been randomly shuffled')
    print()
    print('The total number of tracks to sort through: ' + str(len(tracks)))
    print()
    
    indices_to_keep = []    

    
    output_selection = interactive(select_track_option,
                                   i=widgets.SelectionSlider(description='Track ID:',   options=indices, value = indices[0]),
                                   tracks=fixed(tracks),
                                   indices_to_keep=fixed(indices_to_keep))
    display(output_selection)
    
    
    return output_selection.result

def select_track_option(tracks, i, indices_to_keep):
    """Configure options for manual selection of tracks"""
    selection_widget = widgets.ToggleButton(
                                            value=False,
                                            description='Click to keep',
                                            disabled=False,
                                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                            tooltip='Description',
                                            icon='check'
                                            )
    
    deselection_widget = widgets.ToggleButton(
                                            value=False,
                                            description='Click to remove',
                                            disabled=False,
                                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                            tooltip='Description',
                                            icon='check'
                                            )
    
    output_selection = interactive(iterate_through_selectable_tracks,
                                   selection=selection_widget,
                                   deselection=deselection_widget,
                                   i=fixed(i),
                                   tracks=fixed(tracks),
                                   indices_to_keep=fixed(indices_to_keep))
    display(output_selection)
    
    return output_selection.result

def iterate_through_selectable_tracks(tracks, i, selection, deselection, indices_to_keep):
    """Plot each track for manual track selection"""
    channel_colors = ['m', 'g']
    
    for channel_number in [0,1]:

        plt.plot(return_track_amplitude_one_channel(tracks,
                                                    i,
                                                    channel_number),
                        color=channel_colors[channel_number],
                        label='ch' + str(channel_number))

        plt.xlabel('frames')
        plt.ylabel('au fluorescence intensity')
        plt.legend()
    
    plt.show()

    if selection:
        
        indices_to_keep.append(i)
        
    if deselection:
        
        indices_to_keep.remove(i)
        
    print('selected IDs:' + str(indices_to_keep))
    
    return indices_to_keep
    
    
    
def display_tracks(tracks, 
                   minimum_lifetime=0, 
                   maximum_lifetime=np.Inf, 
                   track_category=[1, 2, 3, 4, 5, 6, 7, 8], 
                   ch_min_int_threshold=[-np.Inf], 
                   number_of_channels=2, 
                   display_channels=[0],
                   interact_through_tracks=True,
                   display_all=False, 
                   only_ccps=False, 
                   display_intensity=True,
                   display_movement=False,
                   display_distance_traveled=False,
                   show_frames=False,
                   create_gif=False,
                   raw_frames=[],
                   display_separation=False,
                   first_separation_channel=0,
                   second_separation_channel=0,
                   constraints_overlap_ccps=False, 
                   dpi=300, 
                   fig_size=(3, 2), 
                   channel_colors=['g']):
    
    """
    Display track features for all tracks in a Python-coverted ProcessedTracks.mat
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        minimum_lifetime (int, optional): minimum lifetime of tracks to display
        maximum_lifetime (int, optional): maximum lifetime of tracks to display 
        track_category (list, optional): category of tracks to display as designated by ProcessedTracks.mat
        ch_min_int_threshold (list, optiona): minimum intensity threshold for each channel necessary to display
        number_of_channels (int, optional): number of channels in tracks
        display_channels (list, optional): channels from tracks to display 
        interact_through_tracks (boolean, optional): display widgets for interactively loading tracks one at a time by track index
        display_all (boolean, optional): display all tracks without any constraints applied
        only_ccps (boolean, optional): display only tracks designated as 'CCPs' by cmeAnalysis
        display_intensity (boolean, optional): display intensity vs. time 
        display_movement (boolean, optional): display (x(t),y(t)) in pixels
        display_distance_traveled (boolean, optional): display distance traveled from origin of track
        show_frames (boolean, optional): show raw data centered around track
        create_gif (boolean, optional): create gif of raw data separated by channels and intensity vs. time for designated chnanels 
        raw_frames (list of strings, optional): list of strings of pathnames for raw data channels (must contain all channels for tracks)        
        display_separation (boolean, optional): display separation between two channels' fitted center vs. time 
        first_separation_channel (int, optional): index of first channel for separation display
        second_separation_channel (int, optional): index of second chnanel for separation display
        contraints_overlap_ccps (boolean, optional): display tracks designated as 'CCPs' and meet designated criteria
        dpi (int, optional): dpi for matplotlib figure
        fig_size (tuple, optional): size of matplotlib figure
        channel_colors (list, optional): colors for plotting traces corresponding to each displayed channel
 

     Returns:
         None

    """
    
    
    number_of_tracks = len(tracks)

    print()
    print('The total number of tracks: ' + str(number_of_tracks))
    print()

    indices_to_display = []
    number_of_ccps_total = 0
    number_of_ccps_condition_bound = 0

    for i in range(0, number_of_tracks):

        if display_all:

            indices_to_display.append(i)


            number_of_ccps_condition_bound += 1


        elif only_ccps:

            if return_is_CCP(tracks, i):

                number_of_ccps_condition_bound += 1

                indices_to_display.append(i)

        elif constraints_overlap_ccps:

            if track_within_bounds(tracks, i, minimum_lifetime, maximum_lifetime, track_category, number_of_channels) and \
               return_is_CCP(tracks, i):

                number_of_ccps_condition_bound += 1

                indices_to_display.append(i)


#         check if current track is within defined bounds
        elif track_within_bounds(tracks, i, minimum_lifetime, maximum_lifetime, track_category):

            num_channels_in_track = len(return_track_amplitude(tracks, i))

            if num_channels_in_track > len(channel_colors):

                raise Exception('length of "channel_colors" must be equal to "number_of_channels"')

            if len(ch_min_int_threshold) == 1:

                temp = 0

                for j in range(num_channels_in_track):

                    if max(return_track_amplitude(tracks, i)[j]) > ch_min_int_threshold[0]:

                        temp += 1

                if temp == num_channels_in_track:

                    number_of_ccps_condition_bound += 1

                    indices_to_display.append(i)

            elif len(ch_min_int_threshold) == num_channels_in_track:

                temp = 0

                for j in range(num_channels_in_track):

                    if max(return_track_amplitude(tracks, i)[j]) > ch_min_int_threshold[j]:

                        temp += 1

                if temp == num_channels_in_track:

                    number_of_ccps_condition_bound += 1

                    indices_to_display.append(i)
            else:

                raise Exception('length of "ch_min_int_threshold" must be equal to maximum number of channels in experiment OR set to default ([0])')


        if return_is_CCP(tracks, i) == 1:

            number_of_ccps_total += 1

    if not display_all:
        print('The total number of ccps identified by cmeAnalysis: ' + str(number_of_ccps_total))
        print()
    print('The total number of tracks to be displayed within defined bounds: ' + str(number_of_ccps_condition_bound))
    print()
    print()

    if interact_through_tracks:
        
        interact(display_one_track,
             tracks=fixed(tracks),
             track_number = widgets.SelectionSlider(description='Track ID:',   options=indices_to_display, value = indices_to_display[0]), 
             dpi=fixed(dpi), 
             fig_size=fixed(fig_size), 
             channel_colors=fixed(channel_colors),
             display_channels=fixed(display_channels),
             display_intensity=fixed(display_intensity),
             display_movement=fixed(display_movement),
             display_distance_traveled=fixed(display_distance_traveled),
             display_separation=fixed(display_separation),
             show_frames=fixed(show_frames),
             create_gif=fixed(create_gif),
             raw_frames=fixed(raw_frames),
             first_separation_channel=fixed(first_separation_channel),
             second_separation_channel=fixed(second_separation_channel))
            
        
    else:
        for track_ID in indices_to_display:

            display_one_track(tracks, 
                              track_ID, 
                              dpi, 
                              fig_size, 
                              display_channels,
                              channel_colors, 
                              display_intensity,
                              display_movement,
                              display_distance_traveled,
                              display_separation,
                              show_frames,
                              create_gif,
                              raw_frames,
                              first_separation_channel,
                              second_separation_channel)
    


def remove_tracks_by_criteria(tracks, 
                              minimum_lifetime=0, 
                              maximum_lifetime=np.Inf, 
                              track_category=[1, 2, 3, 4, 5, 6, 7, 8], 
                              ch_min_int_threshold=[-np.Inf], 
                              keep_all=False, 
                              only_ccps=False, 
                              constraints_overlap_ccps=False):    
    tracks_return = []
    new_index = 0
    for i in range(0, len(tracks)):

        if keep_all:
            

            tracks_return.append(tracks[i])
    
        elif only_ccps:

            if return_is_CCP(tracks, i) == 1:

                tracks_return.append(tracks[i])
                
        elif constraints_overlap_ccps:

            if track_within_bounds(tracks, i, minimum_lifetime, maximum_lifetime, track_category) and \
               return_is_CCP(tracks, i) == 1:
                
                tracks_return.append(tracks[i])



        elif track_within_bounds(tracks, i, minimum_lifetime, maximum_lifetime, track_category):

            num_channels_in_track = len(return_track_amplitude(tracks, i))


            if len(ch_min_int_threshold) == 1:
            
                temp = 0

                for j in range(num_channels_in_track):

                    if max(return_track_amplitude_one_channel(tracks, i, j)) > ch_min_int_threshold[0]:
                        
                        temp += 1

                if temp == num_channels_in_track:

                    tracks_return.append(tracks[i])
    
            elif len(ch_min_int_threshold) == num_channels_in_track:

                temp = 0

                for j in range(num_channels_in_track):

                    if max(return_track_amplitude_one_channel(tracks, i, j)) > ch_min_int_threshold[j]:

                        temp += 1

                if temp == num_channels_in_track:


                    tracks_return.append(tracks[i])
                    
    print('The number of tracks returned: ' + str(len(tracks_return)))
    print()
    return tuple(tracks_return)




def remove_tracks_by_indices(tracks,
                             selected_indices=[]):
    """Return tracks provided sans those whose number IDs are provided in 'selected_indices' """
    
    selected_indices.sort()

    indices_to_keep = (list(set([*range(len(tracks))]).difference(selected_indices))) # find the indices to keep
    indices_to_keep.sort()
    
    tracks_return = select_tracks_by_indices(tracks, indices_to_keep) # select and return the tracks whose IDs are not in 'selected_indices' 

    return tracks_return


def select_tracks_by_indices(tracks,
                             selected_indices=[]):
    """Return tracks whose number IDs are provided by 'selected_indices' """
    
    print()
    print('The total number of tracks: ' + str(len(tracks)))
    print()

    tracks_return = np.array(list(tracks))[selected_indices]

    print('The number of tracks returned: ' + str(len(tracks_return)))       
    print()
    return tracks_return



            
def create_tp_tn_fp_fn_labels(test_labels, predicted_labels):
    """
    Find indices for true positives, false negatives, false positives, and false negativess from ground truth and predicted labels
    
    Args:
        test_labels (list): list of ground-truth labels
        predicted_labels (list): list of predicted labels
        
    Returns:
        predicted_labels (list of lists): lists (in order: tp, tn, fp, fn) of indices of prediction outcomes
        
    """
    
    
    if len(test_labels)!=len(predicted_labels):
        raise Exception('arrays must be equal in size')
    
    tp = []
    tn = []
    fp = []
    fn = []
    
    for i in range(len(test_labels)):
        
        if test_labels[i]==1 and predicted_labels[i]==1:
            tp.append(i)
        elif test_labels[i]==0 and predicted_labels[i]==0:
            tn.append(i)
        elif test_labels[i]==0 and predicted_labels[i]==1:
            fp.append(i)
        elif test_labels[i]==1 and predicted_labels[i]==0:
            fn.append(i)

    return [tp,tn,fp,fn]

def plot_subplots_of_labels(tracks, 
                            number_of_channels, 
                            channel_colors,
                            display_channels,
                            track_indices, 
                            number_of_columns, 
                            num_plot,
                            filename='', 
                            minimum_lifetime = 0,
                            maximum_lifetime = np.Inf,
                            include_background=False):    
    """
    Plot a subplot grid of a subset of random samples from the designated tracks
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        number_of_channels (int): number of channels in tracks object
        channel_colors (list): each entry is a string corresponding to the plotted channels' matplotlib color
        display_channels (list): each entry is an integer corresponding to the channel to be plotted
        track_indices (list): indices of potential tracks for plotting
        number_of_columns (int): number of columns in the subplot (number of rows determined automatically)
        num_plot (int): number of tracks to plot 
        filename (str, optional): pathname of figure to be saved as png
        minimum_lifetime (int, optional): minimum lifetime of tracks to sample
        maximum_lifetime (int, optional): maximum lifetime of tracks to sample 
        include_background (boolean): plot intensities on top of background and also plot background
        
    Returns:
        None
        
    """
 
    subplot_index = 1

    num_rows = int(np.ceil(num_plot/number_of_columns))

    f = figure(figsize=(3*number_of_columns+5,3*num_rows+5),dpi=500)
    
    tracks_to_plot = select_tracks_by_indices(tracks, track_indices)
    tracks_to_plot = remove_tracks_by_criteria(tracks_to_plot, minimum_lifetime=minimum_lifetime, maximum_lifetime=maximum_lifetime)

    random_indices = list(range(len(tracks_to_plot)))
    random.shuffle(random_indices)
    track_indices = random_indices[:num_plot]
    
    tracks_to_plot = select_tracks_by_indices(tracks_to_plot, track_indices)    

    lifetimes = []
    for i in range(len(tracks_to_plot)):

        lifetimes.append(return_track_lifetime(tracks_to_plot,i))
        
    sorted_lifetime_indices = sorted(range(len(lifetimes)), key=lambda k: lifetimes[k], reverse=True)
    
    tracks_to_plot_list = [tracks_to_plot[i] for i in sorted_lifetime_indices][:num_plot]
    



    for i in range(len(tracks_to_plot_list)):
#         print(number_of_columns,num_rows)
        ax = f.add_subplot(num_rows,number_of_columns,subplot_index)
        
        if include_background:
            
            for channel_number in display_channels:
                
                intensity = return_track_amplitude_no_buffer_channel(tracks_to_plot_list, i, channel_number)                
                background = tracks_to_plot_list[i][index_dictionary['index_background']][channel_number]
                background_std = tracks_to_plot_list[i][index_dictionary['index_sigma_r']][channel_number]    
                    
                ax.plot(intensity + background, color=channel_colors[channel_number], label='ch' + str(channel_number))
                ax.fill_between(x=np.arange(len(background)), 
                                y1=background,    
                                y2=background+2*background_std, 
                                color=channel_colors[channel_number], 
                                alpha=0.5)
            ax.set(xlabel='frames', ylabel='au fluorescence intensity')
            ax.legend()

            
        else:
            
            display_intensity_subplot(ax,
                                      tracks_to_plot_list,
                                      i,
                                      channel_colors,
                                      display_channels)
        
        subplot_index+=1
    
    plt.tight_layout()
    
    if filename!='':
        
        f.savefig(filename)
    
    plt.show()
    

def pairgrid_heatmap(x, y, **kws):
    'https://stackoverflow.com/questions/43924280/pair-plot-with-heat-maps-possibly-logarithmic'
    cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
    plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)
    
    
