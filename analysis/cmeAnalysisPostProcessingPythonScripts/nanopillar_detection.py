# Cyna Shirazinejad, edits made 7/28/20, supervised by Bob Cail and David Drubin
# analysis for nanopillar/grid clathrin-mediated endocytosis (cme) project

# import all necessary libraries
import os # manipulating directories and their contents
import sys # designating system dependencies
import itk # multi-channel, multi-frame tiffs
import scipy.io as sio # image manipulation
from numba import jit # speed up of routine mathematical functions
import numpy as np # dealing with numerical array-like structures
import cv2 # computer-vision related tasks
import matplotlib.pyplot as plt # all things plotting
from matplotlib.pyplot import figure, cm 
# from skimage.feature import canny # canny edge detection algorithm
# from skimage import data # standard text images for prototyping
from tqdm import tqdm # progress bars for loops
from PIL import Image # alternative image importing
from skimage.transform import (hough_line, hough_line_peaks,
                                   probabilistic_hough_line) # hough line transformations and line detection
from generate_index_dictionary import return_index_dictionary # indexing attributes associated with cme tracks in each event
from return_track_attributes import * # extracting event attributes from individual cme tracks
from ipywidgets import interact, interactive, fixed, interact_manual # interactive tools for Jupyter
import ipywidgets as widgets 
from ipywidgets import FloatSlider
from sklearn.preprocessing import normalize # normalizing images for processing
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset # for plot decoration


def create_grid_display_nanopillars(bf_image,
                                    frame_fl,
                                    track_path,
                                    frame,
                                    kernel_size,
                                    stds_kept):
    """
    Display subplots showing initial guesses of Hough detection on raw images
    
    Parameters
    ----------
    - bf_image: string
        - path to single frame of brightfield image of nanopillars/grids (must be contrast adjusted RBG tiff)
    - frame_fl: string
        - path to fluorescent channel frame corresponding to bf_image (must be contrast adjusted RBG tiff)
    - track_path: string
        - path to ProcessedTracks.mat output from cmeAnalysis corresponding to bf_image and frame_fl
    - frame: integer
        - frame of fluorescent channel and corresponding tracks to display
    
    Returns
    -------
    none
    """
    
    

    # intialize a 2x2 plot
    f, axes = plt.subplots(2, 2, figsize=(20,20),gridspec_kw = {'wspace':0.1, 'hspace':0.1}, dpi=50)

#     ####################################################################################################
#     ####################################################################################################
#     # make_cell_profiler_overlay (bottom right)

#     # read cell profiler puncta
#     with open(puncta_csv) as f:
#         reader = csv.reader(f)
#         coords = list(reader)

#     # conver to list of tuple pairs
#     punctafloats = [tuple(float(val) for val in sublist) for sublist in coords]

#     num_on_pillar = 0
#     num_off_pillar = 0

#     # load the fluorescent image (multi-chanel so itk works better than cv2)
#     fl_itk_image = load_itk_image(fl_image)

#     axes[1,1].scatter(*zip(*mask_for_templates),color='r',linewidth = 0.1,alpha = 0.005) # plot the mask
#     axes[1,1].imshow(cv2.cvtColor(cv2.imread(bf_image),cv2.COLOR_BGR2GRAY),alpha=0.5,cmap='Greys') # plot the bright-field, first upload with imread(), then convert to greyscale with cvtColor()
#     axes[1,1].imshow(fl_itk_image[frame_fl],alpha=0.8) # show the fluorescent channel
#     # test whether each puncta is on or off the list
#     for pair in punctafloats:
#         on_pillar = test_on_pillar(pair,mask_for_templates,1) # pass in each pair of (x,y), the mask, and a distance (by pixel length) tolerance for points away from the mask [mask is integers, puncta are floats]
#         if on_pillar==1:
#             num_on_pillar+=1
#             axes[1,1].plot(pair[0],pair[1],marker='x',markerSize=3,color='r')
#         else:
#             num_off_pillar+=1
#             axes[1,1].plot(pair[0],pair[1],marker='x',markerSize=3,color='g')
#     axes[1,1].set_title('fluorescent puncta from Cell Profiler')
#     axes[1,1].set_xlabel('pixels')
#     axes[1,1].set_ylabel('pixels')

#     percentage_on_pillar_cell_profiler = num_on_pillar/(num_on_pillar+num_off_pillar)*100
#     print('The percentage of puncta on nanopillars predicted by Cell Profiler: ' + str(percentage_on_pillar_cell_profiler))

#     # take the bottom right plot, zoom in by 2.5, put in 3rd quadrant
#     axins = zoomed_inset_axes(axes[1,1], 2.5, loc=3) 


#     # add mask to inset
#     axins.scatter(*zip(*mask_for_templates),color='r',linewidth = 0.01,alpha = 0.05)
#     # add brightfield to inset
#     axins.imshow(cv2.cvtColor(cv2.imread(bf_image),cv2.COLOR_BGR2GRAY),alpha=0.5,cmap='Greys')
#     # add puncta to inset
#     for pair in punctafloats:
#         on_pillar = test_on_pillar(pair,mask_for_templates,1)
#         if on_pillar==1:
#             axins.plot(pair[0],pair[1],marker='x',markerSize=3,color='r')
#         else:
#             axins.plot(pair[0],pair[1],marker='x',markerSize=3,color='g')

#     x1, x2, y1, y2 = 400, 500, 100, 200 # specify the limits
#     axins.set_xlim(x1, x2) # apply the x-limits
#     axins.set_ylim(y1,y2) # apply the y-limits
#     plt.yticks(visible=False)
#     plt.xticks(visible=False)
#     axins.invert_yaxis()
#     # add two placeholder x's in regions outside of the cropped inset to designate the labels for the legend
#     axins.scatter(0,0,marker='x',linewidths=3,color='r',label='on pillar')
#     axins.scatter(0,0,marker='x',linewidths=3,color='g', label='off pillar')
#     _, corners1,corners2=mark_inset(axes[1,1], axins, loc1=2, loc2=4, fc="none", ec="1")
#     corners1.loc1 = 1
#     corners1.loc2 = 3
#     corners2.loc1 = 4
#     corners2.loc2 = 2
#     axins.legend()
#     ####################################################################################################
#     ####################################################################################################




    ####################################################################################################
    ####################################################################################################
    # make raw image
    axes[0,0].imshow(cv2.imread(bf_image),alpha=1,cmap='Blues')
    axes[0,0].set_title('raw brightfield image')
    axes[0,0].set_xlabel('pixels')
    axes[0,0].set_ylabel('pixels')
    ####################################################################################################
    ####################################################################################################




    ####################################################################################################
    ####################################################################################################
    # make hough lines
    th2, gray = format_image_hough_detection(bf_image,5) #th2 is thresholded image, gray is grayscale image
    rows, cols = gray.shape # get size of image
    axes[0,1].imshow(gray) # show the brightfield image
    y0s, y1s, slopes = hough_detection(bf_image, kernel_size, stds_kept, show_slopes=False)
    for i in range(len(y0s)):
        # plot all hough lines
        axes[0,1].plot((0, cols), (y0s[i], y1s[i]), '-r')

    axes[0,1].axis((0, cols, rows, 0)) # set the axis marks

    axins = zoomed_inset_axes(axes[0,1], 2.5, loc=1) # create an inset in quadrant 1
    axins.imshow(gray) # add brightfield to inset
    for i in range(len(y0s)):
        # add hough lines to inset
        axins.plot((0, cols), (y0s[i], y1s[i]), '-r')
    x1, x2, y1, y2 = 75, 175, 100, 200 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1,y2) # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    axins.invert_yaxis()
    # specify the corners inset and highlighted region are connected
    _, corners1,corners2=mark_inset(axes[0,1], axins, loc1=2, loc2=4, fc="none", ec="1")
    corners1.loc1 = 2
    corners1.loc2 = 4
    corners2.loc1 = 3
    corners2.loc2 = 1
    axes[0,1].set_title('detected Hough lines')
    axes[0,1].set_xlabel('pixels')
    axes[0,1].set_ylabel('pixels')
    ####################################################################################################
    ####################################################################################################



    ####################################################################################################
    ####################################################################################################
    # make cmeAnalysis plot
    # add mask
    
    mask = create_mask_from_hough_lines(bf_image,y0s,y1s,2)
    
    axes[1,0].scatter(*zip(*mask),color='r',linewidth = 0.1,alpha = 0.005)
    # add fluorescence channel
    axes[1,0].imshow(fl_itk_image[frame],alpha=1)
    # load track object from pathname
    tracks = load_tracks(track_path)
    
    # extract coordinates of puncta on and off pillar
    coordinates_of_points_x_on_pillar, coordinates_of_points_y_on_pillar, \
    coordinates_of_points_x_off_pillar, coordinates_of_points_y_off_pillar = return_puncta_relative_to_pillar(tracks, 0, 0, mask_for_templates)

    percentage_on_pillar_cmeAnalysis = len(coordinates_of_points_x_on_pillar)/(len(coordinates_of_points_x_on_pillar)+ len(coordinates_of_points_x_off_pillar))*100
    print('The percentage of puncta on nanopillars predicted by cmeAnalysis: ' + str(percentage_on_pillar_cmeAnalysis))

    # plot puncta on pillar
    axes[1,0].scatter(coordinates_of_points_x_on_pillar,coordinates_of_points_y_on_pillar,marker='x',linewidths=0.1,alpha=0.7,color='r')
    # plot puncta off pillar
    axes[1,0].scatter(coordinates_of_points_x_off_pillar,coordinates_of_points_y_off_pillar,marker='x',linewidths=0.1,alpha=0.7,color='g')
    axes[1,0].set_title('fluorescent puncta from cmeAnalysis')
    axes[1,0].set_xlabel('pixels')
    axes[1,0].set_ylabel('pixels')
    # create inset
    axins = zoomed_inset_axes(axes[1,0], 2.5, loc=3) # zoom-factor: 2.5, location: upper-left
    # add mask to inset
    axins.scatter(*zip(*mask),color='r',linewidth = 0.01,alpha = 0.05)
    # add fluorescence channel to inset
    axins.imshow(fl_itk_image[frame_fl],alpha=1)
    # add puncta to inset
    axins.scatter(coordinates_of_points_x_on_pillar,coordinates_of_points_y_on_pillar,marker='x',linewidths=0.1,alpha=0.7,color='r')
    axins.scatter(coordinates_of_points_x_off_pillar,coordinates_of_points_y_off_pillar,marker='x',linewidths=0.1,alpha=0.7,color='g')
    x1, x2, y1, y2 = 400, 500, 100, 200 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1,y2) # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    axins.invert_yaxis()
    axins.scatter(0,0,marker='x',linewidths=1,color='r',label='on pillar')
    axins.scatter(0,0,marker='x',linewidths=1,color='g', label='off pillar')
    _, corners1,corners2=mark_inset(axes[1,0], axins, loc1=2, loc2=4, fc="none", ec="1")
    corners1.loc1 = 1
    corners1.loc2 = 3
    corners2.loc1 = 4
    corners2.loc2 = 2
    axins.legend()
    ####################################################################################################
    ####################################################################################################
    # save the plot and set high dpi
#     plt.savefig('photos_of_nanopillar_figures/sample_for_bob.png',dpi=500)

def modify_filename(file_name, new_identifier):
    """
    Return a modified file name for saving finalized mask objects
    
    The provided file name will be truncated after the last forward
    slash: '/'. This will first be appended by the provided string after 
    the last forward slash. Finally, the returned string will then be 
    appended by the characters following the last forward slash leading 
    up to the first period '.' following the last forward slash.
    
    Parameters
    ----------
    - file_name : string
        - template string of the full path to the bright-field image in the 
          parent directory of the experimental condition.
    - new_identifier : string
        - unique identifier descriptive of the saved output i.e. 'mask_'
    
    Returns
    -------
    string
        The returned full path of the mask object to be saved.
    """
    file_split = file_name.split('/') # 
    new_string = ''
    for idx in file_split[:-1]:
        new_string += '/'
        new_string += idx
    new_string += '/'
    last_file_identifier = file_split[-1]
    last_file_minus_extra = last_file_identifier.split('.')[0]
    new_string += new_identifier
    new_string += last_file_minus_extra
    new_string = new_string[1:]
    return new_string

def format_image_hough_detection(file_location,smoothing_kernel_size):
    """
    Format raw brightfield image for Hough detection
    
    Parameters
    ----------
    - file_location : string
        - Path to brightfield image
    - smoothing_kernel_size: int
        - Size of nxn Gaussian kernel used for smoothing image (must be odd number)
    
    Returns
    -------
    - th2: array
        - thresholded raw image with shape of raw image that has been adaptively thresholded
    - gray: array
        - gray-scale image converted from raw image 
    """    
    # import the image, it is a 3 channel RBG Tiff Brightfield image from Fiji 
    img = cv2.imread(file_location)
    
    filename = file_location.split('/')[-1]
    print('File being segmented: ' + '\n' + str(filename))

    # invert image colors black to white and vice versa
    
    img = cv2.bitwise_not(img)
    
    # convert image to gray from RBG
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Gaussian smoothing with (N,N) pixel kernel
    gray = cv2.GaussianBlur(gray,(smoothing_kernel_size,smoothing_kernel_size),0)

    # adaptive thresholding to smooth out image with (possible) uneven illumination
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,3,0)
    
    return th2, gray

def hough_detection(file_location, smoothing_kernel_size, std_fraction, show_slopes, min_slope=-np.Inf, max_slope=np.Inf):
    """
    Detect Hough lines in formatted brightfield image of nanopillars/grids
    
    Parameters
    ----------
    - file_location : string
        - path to brightfield image
    - smoothing_kernel_size: int
        - size of nxn Gaussian kernel used for smoothing image (must be odd number)
    - std_fraction: float
        - number of standard deviations around the mean of slopes of detected lines to keep and return
    - show_slopes: boolean
        - plot and print the accepted slopes
    Returns
    -------
    - [y0s, y1s, slopes]: array of lists
        - the end points falling along x={0,x_max} and slopes of each Hough line
    
    """
    
    th2, gray = format_image_hough_detection(file_location,smoothing_kernel_size)

    # Hough transform (find lines)
    h, theta, d = hough_line(th2)

    
    # generating figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), dpi=500)
    plt.tight_layout()

    ax0.imshow(th2, cmap='gray')
    ax0.set_title('Input image')
    ax0.set_axis_off()

    ax1.imshow(gray, cmap=cm.gray)
    row1, col1 = gray.shape
    print('the size of image is :' + str((row1,col1)))
    slopes = []
    y1s = []
    y0s = []
    # find line end points for plotting
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
        slopes.append((y0-y1)/(0-col1))
        y1s.append(y1)
        y0s.append(y0)

    # calculate slope statistics
    average_slope = np.mean(slopes)
    std_slope = np.std(slopes)

    # acceptable slope bounds 
    lower_bound_slopes = average_slope - std_fraction*std_slope
    upper_bound_slopes = average_slope + std_fraction*std_slope

    if min_slope != -np.Inf:
        
        lower_bound_slopes = min_slope
    
    if max_slope != np.Inf:
        
        upper_bound_slopes = max_slope
    
    accepted_slope_indices = [i for i in range(len(slopes)) if slopes[i] > lower_bound_slopes and slopes[i] < upper_bound_slopes]
    
    y0s_accepted=[]
    y1s_accepted=[]
    slopes_accepted = []
    for i in range(len(slopes)):

        if i in accepted_slope_indices:

            y0s_accepted.append(y0s[i])
            y1s_accepted.append(y1s[i])
            slopes_accepted.append((y0s[i]-y1s[i])/(0-col1))
    for i in range(len(y1s_accepted)):

        ax1.plot((0, col1), (y0s_accepted[i], y1s_accepted[i]), '-r')

    
    ax1.axis((0, col1, row1, 0))
    ax1.set_title('Detected lines')
    ax1.set_axis_off()
    y0s_accepted=np.array(y0s_accepted)
    y1s_accepted=np.array(y1s_accepted)
    slopes_accepted=np.array(slopes_accepted)
    inds = y0s_accepted.argsort()
    y0s_accepted=y0s_accepted[inds]
    y1s_accepted=y1s_accepted[inds]
    slopes_accepted=slopes_accepted[inds]
    
    average_slope = np.mean(slopes_accepted)
    std_slope = np.std(slopes_accepted)
    
    return_values = [y0s_accepted, y1s_accepted, slopes_accepted]

    for i in range(len(return_values[0])):

        plt.text(10,return_values[0][i],str(i))
        plt.text(10,return_values[0][i],str(i))
        plt.text(10+30,return_values[0][i],str(np.around(return_values[2][i],2)))
        
    plt.show()
    
    if show_slopes:
        print('The slopes kept are: '+ str(slopes_accepted))
        print('average slope: ' + str(average_slope))
        print('std of slopes: ' + str(std_slope))
        plt.xlim([0,np.max(slopes_accepted)])
        plt.hist(slopes_accepted,bins=len(slopes_accepted))
        plt.title('all line slopes (use this to chose std # cut-off)')
        plt.xlabel('slope')
        plt.ylabel('count')
        plt.show()

        average_slope_kept = np.mean(slopes_accepted)
        print('average of kept slopes: ' + str(average_slope_kept))

    
    return return_values[0], return_values[1], return_values[2]




def createLineIterator(P1, P2, img):
    # https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator

    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -itbuffer: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
   #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

def create_mask_from_hough_lines(filename,y0s,y1s,vertical_tolerance):
    """
    Create a mask from provided Hough line outputs
    
    Parameters
    ----------
    -filename: string
        - path to brightfield of nanopillars
    - y0s: list of floats
        - y values of Hough lines at x=0
    - y1s: list of floats
        - y values of Hough lines at x=x_max
    - vertical_tolerance: integer
        - number of pixels to spread Hough lines over above and below each pixel connecting each pair of y0 and y1
    
    Returns
    -------
    - mask: array
        - array of size of original image with 1s where Hough lines exist with 0 elsewhere
        
    """

    
    figure(num=None, figsize=(12, 12), dpi=100, facecolor='w', edgecolor='k')
    
    y0s = np.sort(y0s)
    y1s = np.sort(y1s)
    img = cv2.imread(filename)


    filename = filename.split('/')[-1]
    print('File being segmented: ' + '\n' + str(filename))

    # invert image colors black to white and vice versa

    img = cv2.bitwise_not(img)

    # convert image to gray from RBG
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Gaussian smoothing with (N,N) pixel kernel
    gray = cv2.GaussianBlur(gray,(3,3),0)
    plt.imshow(gray, cmap='gray')
    row1, col1 = gray.shape
    
#     masked_positions = []
    mask = np.zeros((row1,col1))
    positions = []
    for i in tqdm(range(len(y0s))):
    
        pixel_left = np.int(y0s[i])
        pixel_right = np.int(y1s[i])

        pixel_left_range = np.arange(pixel_left-vertical_tolerance,pixel_left+vertical_tolerance+1,step=1)
        pixel_right_range = np.arange(pixel_right-vertical_tolerance,pixel_right+vertical_tolerance+1,step=1)

        for index in range(len(pixel_left_range)):

            output_line = createLineIterator(np.array([0,pixel_left_range[index]]),np.array([col1,pixel_right_range[index]]), gray)

            for output in output_line:
                positions.append([output[0],output[1]])
                mask[np.int(output[1]),np.int(output[0])] = 1
    plt.imshow(mask,alpha=0.2,cmap='Reds')
# #     positions_scatter_plot = [tuple(l) for l in positions]
#     positions = np.array(positions)
#     print(positions)
#     mask_array = np.zeros(gray.shape)
#     mask_array[tuple(np.array(positions).T)]=1
#     print('test')
# #     mask_array[tuple(positions)]=1
# #     for i in range(row1):
# #         for j in range(col1):
# #             if (i,j) in positions_scatter_plot:
# #                 mask_array[i,j]=1
#     plt.imshow(mask_array)
    
#     plt.scatter(*zip(*positions_scatter_plot),color='r',linewidth = 0.1,alpha = 0.05)
    
    for i in range(len(y0s)):
        
        plt.text(10,y0s[i],str(i),fontweight='bold')
        plt.text(512,y1s[i],str(i),fontweight='bold')
    
    return mask
    
# def shift_endpoints(end_points, shift):
#     new_endpoints = []
#     for pt in end_points:
#         new_endpoints.append(pt+shift)
#     return new_endpoints

# def show_frame_overlay_puncta(frame, bf_image_path, fl_image_path, track_path, channel_number, non_zero):
    
    
#     figure(figsize=(16, 16), dpi=100)

#     fl_itk_image = load_itk_image(fl_image_path)
#     print('test')
#     plt.imshow(fl_itk_image[frame])
# #     plt.imshow(load_itk_image(bf_image_path))
#     bf_itk_image = load_itk_image(bf_image_path)
    
#     tracks = return_track_dictionary(track_path)
    
#     mask_matrix = np.array(non_zero)
                
#     coordinates_of_points_x_on_pillar, coordinates_of_points_y_on_pillar, \
#     coordinates_of_points_x_off_pillar, coordinates_of_points_y_off_pillar = return_puncta_relative_to_pillar(tracks, frame, channel_number, mask_matrix)
    
#     plt.scatter(coordinates_of_points_x_on_pillar,coordinates_of_points_y_on_pillar,marker='x',linewidths=0.1,alpha=0.7,color='r')
#     plt.scatter(coordinates_of_points_x_off_pillar,coordinates_of_points_y_off_pillar,marker='x',linewidths=0.1,alpha=0.7,color='g')
        
#     plt.scatter(*zip(*non_zero),color='y',alpha=0.01)
    


def test_on_pillar(puncta_coordinate,mask_positions,tolerance):
    """ Check if a puncta coordinate is on a mask pixel within a defined tolerance distance"""

    nonzero_mask_positions = np.nonzero(mask_positions)
    B = nonzero_mask_positions[0]
    A = nonzero_mask_positions[1]
    X = np.sqrt( np.square( A - puncta_coordinate[0] ) +  np.square( B - puncta_coordinate[1] ) )
    idx = np.where( X == X.min() )
  
    if (X[idx]<tolerance).any():
        return 1
    else:
        return 0


def load_itk_image(image_path):
    """Loads and returns a multiframe fluorescent image"""
    image = itk.imread(image_path)

    image = itk.array_view_from_image(image)
    
    return image

def return_track_dictionary(track_path):
    """Return processed tracks from cmeAnalysis as a Python array from a full path to ProcessedTracks.mat"""
    tracks = sio.loadmat(track_path)

    index_dictionary = return_index_dictionary()
        # get just the data for the tracks
    tracks = tracks['tracks'][0]
    # # save each track with its original index
    tracks = zip(tracks, range(len(tracks)))
        # sort the tracks in descending lifetime, preserving each track's individual index
    tracks = sorted(tracks, key=lambda track: track[0][index_dictionary['index_lifetime_s']][0][0], reverse=True)
    (tracks, indices) = zip(*tracks)
    
    return tracks

def return_puncta_relative_to_pillar(tracks, frame, channel_number, mask_matrix):
    """Return the puncta positions separated by ones on/near mask pixels and ones off mask pixels"""
    coordinates_of_points_x_on_pillar = []
    coordinates_of_points_y_on_pillar = []
    coordinates_of_points_x_off_pillar = []
    coordinates_of_points_y_off_pillar = []
    
    from generate_index_dictionary import return_index_dictionary
    index_dictionary = return_index_dictionary()
    
    for i in tqdm(range(len(tracks))): 

        if check_puncta_in_frame(tracks, frame, i, channel_number, index_dictionary) and return_track_category(tracks, i) in [1,2,3,4]:
                #checking whether the track belongs in this frame
            puncta_x_position = return_puncta_x_position(tracks,i,channel_number,frame)
            puncta_y_position = return_puncta_y_position(tracks,i,channel_number,frame)
            
            on_pillar = test_on_pillar((puncta_x_position-0.5, puncta_y_position-0.5), mask_matrix, 1)
            
            if on_pillar == 1:
                
                coordinates_of_points_x_on_pillar.append(puncta_x_position-0.5)
                coordinates_of_points_y_on_pillar.append(puncta_y_position-0.5)
            else:
                coordinates_of_points_x_off_pillar.append(puncta_x_position-0.5)
                coordinates_of_points_y_off_pillar.append(puncta_y_position-0.5)
    
    return coordinates_of_points_x_on_pillar, coordinates_of_points_y_on_pillar, coordinates_of_points_x_off_pillar, coordinates_of_points_y_off_pillar
                            
def return_valid_on_pillar(tracks, frame, channel_number, mask_matrix):
    valids_type1_on = []
    valids_type2_on = []
    valids_type3_on = []
    valids_type4_on = []
    valids_type1_off = []
    valids_type2_off = []
    valids_type3_off = []
    valids_type4_off = []
        
    from generate_index_dictionary import return_index_dictionary
    index_dictionary = return_index_dictionary()
        
    for i in range(len(tracks)):
        if check_puncta_in_frame(tracks, frame, i, channel_number, index_dictionary) and return_track_category(tracks, i) in [1,2,3,4]:
                #checking whether the track belongs in this frame and is type 1-4
            puncta_x_position = return_puncta_x_position(tracks,i,channel_number,frame)
            puncta_y_position = return_puncta_y_position(tracks,i,channel_number,frame)
            
            on_pillar = test_on_pillar((puncta_x_position-0.5, puncta_y_position-0.5), mask_matrix, 1)
            
            if on_pillar == 1:
                
                if return_track_category(tracks, i) == 1:
                    valids_type1_on.append(i)
                elif return_track_category(tracks, i) == 2:
                    valids_type2_on.append(i)
                elif return_track_category(tracks, i) == 3:
                    valids_type3_on.append(i)
                else:
                    valids_type4_on.append(i)
            else:
                if return_track_category(tracks, i) == 1:
                    valids_type1_off.append(i)
                elif return_track_category(tracks, i) == 2:
                    valids_type2_off.append(i)
                elif return_track_category(tracks, i) == 3:
                    valids_type3_off.append(i)
                else:
                    valids_type4_off.append(i)
    return valids_type1_on, valids_type2_on, valids_type3_on, valids_type4_on, valids_type1_off, valids_type2_off, valids_type3_off, valids_type4_off



def check_puncta_in_frame(tracks, frame, i, channel_number, index_dictionary):
    """Check if a track belongs in a designated frame of a movie"""
    if (frame+1 in tracks[i][index_dictionary['index_frames']][0]) and \
       channel_number < len(tracks[i][index_dictionary['index_x_pos']]) and \
       len(tracks[i][index_dictionary['index_x_pos']][channel_number]) > frame: 
        
        return True
    
    else:
        return False
    
# def display_raw_data_and_tracks(tracks,
#                                 raw_images):
    
#     interact(interact_through_tracks_IDs,tracks=fixed(tracks),raw_images=fixed(raw_images),track_number=widgets.IntSlider(min=0,max=len(tracks)-1,step=1,value=0))

# def interact_through_tracks_IDs(tracks,raw_images,track_number):
#     print('The track being observed: {}'.format(track_number))
#     print(return_track_amplitude_one_channel(tracks,track_number,0))
#     len_current_track = len(return_track_amplitude_one_channel(tracks,track_number,0))
#     print('The length of the current track: {}'.format(len_current_track))
#     interact(interact_through_frames_of_track,raw_images=fixed(raw_images),track_number=fixed(track_number),tracks=fixed(tracks),frames=widgets.IntSlider(min=0,max=len_current_track-1,step=1,value=0))

# def interact_through_frames_of_track(tracks,raw_images, track_number,frames):
# #     print(return_frames_in_track(tracks,track_number)-1)
#     num_channels = len(return_track_amplitude(tracks,track_number))
#     fig=plt.figure(num=None, figsize=(10,5), dpi=200, facecolor='w', edgecolor='k')
#     index_subplot = 1 
#     colors=['Reds','Greens','Blues']
#     track_x_positions = []
#     track_y_positions = []
#     frames_in_track = return_frames_in_track(tracks,track_number)-1
#     print(frames_in_track)
#     min_x = np.min(return_puncta_x_position_whole_track(tracks,track_number,0))
#     max_x = np.max(return_puncta_x_position_whole_track(tracks,track_number,0))
#     min_y = np.min(return_puncta_y_position_whole_track(tracks,track_number,0))
#     max_y = np.max(return_puncta_y_position_whole_track(tracks,track_number,0))
#     for i in range(num_channels):
#         print('test')
#         print(track_number)
#         track_x_positions.append(return_puncta_x_position_whole_track(tracks,track_number,i)-0.5)
#         track_y_positions.append(return_puncta_y_position_whole_track(tracks,track_number,i)-0.5)
#         if np.min(return_puncta_x_position_whole_track(tracks,track_number,i))<min_x:
#             min_x = np.min(return_puncta_x_position_whole_track(tracks,track_number,i))
#         if np.max(return_puncta_x_position_whole_track(tracks,track_number,i))>max_x:
#             max_x = np.max(return_puncta_x_position_whole_track(tracks,track_number,i))
#         if np.min(return_puncta_y_position_whole_track(tracks,track_number,i))<min_y:
#             min_y = np.min(return_puncta_y_position_whole_track(tracks,track_number,i))
#         if np.max(return_puncta_y_position_whole_track(tracks,track_number,i))>max_y:
#             max_y = np.max(return_puncta_y_position_whole_track(tracks,track_number,i))
#     diff_x = max_x-min_x
#     diff_y = max_y-min_y
#     avg_x = (max_x+min_x)/2
#     avg_y = (max_y+min_y)/2
#     diff_greatest = np.max([diff_x,diff_y])/2+5

#     for i in range((num_channels)):
#         print(np.array(load_itk_image(raw_images[i]).shape))
#         frame = load_itk_image(raw_images[i])[frames_in_track[frames],:,:]
#         frame_norm = normalize(frame, axis=1, norm='l2')
#         ax_current = fig.add_subplot(1,num_channels+1,index_subplot)
#         ax_current.imshow(frame_norm,cmap=colors[i])
#         ax_current.scatter(track_x_positions[i][frames],track_y_positions[i][frames],marker='x')
#         print(len(track_x_positions[i]))
#         print(track_x_positions[i][frames],track_y_positions[i][frames])
#         ax_current.set_xlim([avg_x-diff_greatest,avg_x+diff_greatest])
#         ax_current.set_ylim([avg_y-diff_greatest,avg_y+diff_greatest])

#         print([avg_x-diff_greatest,avg_x+diff_greatest])
#         print([avg_y-diff_greatest,avg_y+diff_greatest])
#         index_subplot+=1 
        
        
#     ax_overlay = fig.add_subplot(1,num_channels+1,num_channels+1)    
    
#     for i in range((num_channels)):
        
#         ax_overlay.imshow(load_itk_image(raw_images[i])[frames],cmap=colors[i],alpha=0.6)
#         ax_overlay.scatter(track_x_positions[i][frames]/(max_x+10-min_x-10),track_y_positions[i][frames]/(max_y+10-min_y-10),marker='x',alpha=0.3)
#         ax_overlay.set_xlim([np.int(min_y)-10,np.int(max_y)+10])
#         ax_overlay.set_ylim([np.int(min_x)-10,np.int(max_x)+10])

#     return frames