import numpy as np
import pandas as pd
import return_track_attributes 
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy import stats
from statsmodels.stats.diagnostic import anderson_statistic

def find_single_peaks_using_existing_model(path_outputs,
                                           dataframe_name,
                                           track_name):
    
    analysis_metadata = np.load(path_outputs+'/dataframes/analysis_metadata.npy', allow_pickle=True)    

    df_merged_features = pd.read_csv(path_outputs+'/dataframes/'+dataframe_name+'.zip')
    df = df_merged_features
    index_DNM2positive = analysis_metadata.item().get('index_DNM2positive')
    number_of_track_splits = analysis_metadata.item().get('number_of_track_splits')
    number_of_clusters = analysis_metadata.item().get('number_of_clusters')

    
    
    
    
    
    merged_all_valid_tracks = np.load(path_outputs+
                                      '/dataframes/'+track_name+'_'+str(0)+'.npy', allow_pickle=True)

    for i in range(1,number_of_track_splits):

        merged_all_valid_tracks = np.concatenate((merged_all_valid_tracks,
                                                 np.load(path_outputs+
                                                         '/dataframes/'+track_name+'_'+
                                                         str(i)+'.npy', 
                                                 allow_pickle=True)))
        
    gmm_class_indices = []

    for i in range(number_of_clusters):

        gmm_class_indices.append(df[df['gmm_predictions']==i].index.values)    

    print('done')
    
    # keep candidate ccp class mixed with hot-spots
    tracks_dnm2_positive = merged_all_valid_tracks[gmm_class_indices[index_DNM2positive]]

    all_dnm2_signal = []

    for i in range(len(tracks_dnm2_positive)): # stack all DNM2 intensities

        raw_dnm2_intensity = list(return_track_attributes.
                                  return_track_amplitude_no_buffer_channel
                                  (tracks_dnm2_positive,i,1))

        all_dnm2_signal += raw_dnm2_intensity    
        
        
    dt = 1 # sampling interval

    FFT = rfft(all_dnm2_signal) # real FFT of DNM2 signals

    # Getting the related frequencies
    freqs = rfftfreq(len(all_dnm2_signal), dt)
    
    sos = signal.butter(4, 0.2, 'lp', fs=1, output='sos') # low-pass 4-th order Butterworth filter

    filtered_amplitudes = [] # filtered DNM2 traces per track of interest

    for i in range(len(tracks_dnm2_positive)):

        raw_intensity = return_track_attributes.return_track_amplitude_no_buffer_channel(tracks_dnm2_positive,i,1)

        # add zeros to end to account for phase shift of near-track-end peaks
        filtered_amplitudes.append(list(list(signal.sosfilt(sos, raw_intensity)) + [0, 0, 0, 0, 0])) 

    current_param_outputs = [] # one-hot encoding of indices of tracks with a single peak (0: multiple peaks)


    for i in range(len(filtered_amplitudes)): # iterate through all filtered amplitudes

        pvals_dnm2 = return_track_attributes.return_pvals_detection_no_buffer(tracks_dnm2_positive, i, 1)

        # measure whether there is 1 peak with the specified peak-finding parameters
        if len(signal.find_peaks(filtered_amplitudes[i], 
                                 distance=analysis_metadata.item()['distance_best_fit'], 
                                 height=analysis_metadata.item()['height_best_fit'],
                                 width=analysis_metadata.item()['width_best_fit'])[0])==1 and len(np.where(np.array(pvals_dnm2)<0.01)[0])>0:

            current_param_outputs.append(1)

        else:

            current_param_outputs.append(0)
            
    indices_dnm2_positive = gmm_class_indices[index_DNM2positive]
    
    indices_dnm2_positive_and_ccp = indices_dnm2_positive[np.where(np.array(current_param_outputs)==1)[0]]
    
    ccp_status_all_tracks = np.zeros(len(merged_all_valid_tracks))
    ccp_status_all_tracks[indices_dnm2_positive_and_ccp] = 1
    df['ccp_status'] = ccp_status_all_tracks
    print('saving dataframe...\n')
    # save the dataframe for subsequent notebooks
    compression_opts = dict(method='zip',
                            archive_name=path_outputs+'/dataframes/'+dataframe_name+'.csv')  

    df.to_csv(path_outputs+'/dataframes/'+dataframe_name+'.zip', index=False,
                                                              compression=compression_opts) 
    
    print('done\n')
    
    return df
    
def identify_single_peaked_dnm2_events(path_outputs):
    
    analysis_metadata = np.load(path_outputs+'/dataframes/analysis_metadata.npy', allow_pickle=True)
    df_merged_features = pd.read_csv(path_outputs+'/dataframes/df_merged_features.zip')
    feature_units = analysis_metadata.item().get('feature_units')
    index_DNM2positive = analysis_metadata.item().get('index_DNM2positive')
    number_of_track_splits = analysis_metadata.item().get('number_of_track_splits')
    number_of_clusters = analysis_metadata.item().get('number_of_clusters')
    
    df = df_merged_features
    
    # load all valid tracks
    merged_all_valid_tracks = np.load(path_outputs+
                                      '/dataframes/merged_all_valid_tracks_0.npy', allow_pickle=True)

    for i in range(1,number_of_track_splits):

        merged_all_valid_tracks = np.concatenate((merged_all_valid_tracks,
                                                 np.load(path_outputs+
                                                         '/dataframes/merged_all_valid_tracks_'+
                                                         str(i)+'.npy', 
                                                 allow_pickle=True)))
        
    gmm_class_indices = []

    for i in range(number_of_clusters):

        gmm_class_indices.append(df[df['gmm_predictions']==i].index.values)    

    print('done')
    
    # keep candidate ccp class mixed with hot-spots
    tracks_dnm2_positive = merged_all_valid_tracks[gmm_class_indices[index_DNM2positive]]

    all_dnm2_signal = []

    for i in range(len(tracks_dnm2_positive)): # stack all DNM2 intensities

        raw_dnm2_intensity = list(return_track_attributes.
                                  return_track_amplitude_no_buffer_channel
                                  (tracks_dnm2_positive,i,1))

        all_dnm2_signal += raw_dnm2_intensity    
        
        
    dt = 1 # sampling interval

    FFT = rfft(all_dnm2_signal) # real FFT of DNM2 signals

    # Getting the related frequencies
    freqs = rfftfreq(len(all_dnm2_signal), dt)
    
    sos = signal.butter(4, 0.2, 'lp', fs=1, output='sos') # low-pass 4-th order Butterworth filter

    filtered_amplitudes = [] # filtered DNM2 traces per track of interest

    for i in range(len(tracks_dnm2_positive)):

        raw_intensity = return_track_attributes.return_track_amplitude_no_buffer_channel(tracks_dnm2_positive,i,1)

        # add zeros to end to account for phase shift of near-track-end peaks
        filtered_amplitudes.append(list(list(signal.sosfilt(sos, raw_intensity)) + [0, 0, 0, 0, 0])) 
        
    vectorized_find_single_peak_events_dist_height_width = np.vectorize(find_single_peak_events_dist_height_width, excluded=[3,4])

    distances_3 = np.arange(1,20,1) # minimum peak to peak distances
    heights_3 = np.arange(50,300,25) # minimum peak heights
    widths_3 = np.arange(1,10,1) # minimum peak widths

    # create a 3-D mesh of all possible combinations of peak-finding parameters
    distances_mesh_3, heights_mesh_3, widths_mesh_3 = np.meshgrid(distances_3,
                                                                  heights_3, 
                                                                  widths_3,
                                                                  indexing='ij')
#     print(len(filtered_amplitudes))
#     print(len(tracks_dnm2_positive))
#     print('test')
    a, b, c, d, e, f, g = vectorized_find_single_peak_events_dist_height_width(distances_mesh_3, 
                                                                               heights_mesh_3,
                                                                               widths_mesh_3,
                                                                               filtered_amplitudes, 
                                                                               tracks_dnm2_positive)
#     print(len(filtered_amplitudes))
#     print(len(tracks_dnm2_positive))
    num_single_peaked_with_width = a
    significance_position_with_width = b
    chi_squared_gof_stat = c
    chi_squared_gof_pval = d
    sse_param_sweep = e
    peak_detections_with_widths = f
    number_of_peaks_in_event_per_model = g
    
#     all_peak_predictions_with_widths = []

#     for combo in peak_detections_with_widths.items():

#         all_peak_predictions_with_widths.append(peak_detections_with_widths.item()(combo))

#     all_peak_predictions_with_widths = np.array(all_peak_predictions_with_widths).T

#     df_peak_predictions_with_widths = pd.DataFrame(data=all_peak_predictions_with_widths, columns=peak_detections_with_widths.keys())

    ind_best_fit = np.unravel_index(np.argmax(significance_position_with_width, axis=None), significance_position_with_width.shape)

    distance_best_fit = distances_3[ind_best_fit[0]]
    height_best_fit = heights_3[ind_best_fit[1]]
    width_best_fit = widths_3[ind_best_fit[2]]
#     track_indices_best_fit_model_single_peak = peak_detections_with_widths.item().get('min_dist_'+
#                                                                            str(distance_best_fit)+
#                                                                            '_min_height_'+
#                                                                            str(height_best_fit)+
#                                                                            '_min_width_'+str(width_best_fit))
    input_key = 'min_dist_'+str(distance_best_fit)+'_min_height_'+str(height_best_fit)+'_min_width_'+str(width_best_fit)
    
    
    peak_predictions_best_model = peak_detections_with_widths[ind_best_fit][input_key]
    
    analysis_metadata.item()['distance_best_fit'] = distance_best_fit
    analysis_metadata.item()['height_best_fit'] = height_best_fit
    analysis_metadata.item()['width_best_fit'] = width_best_fit
    analysis_metadata.item()['peak_predictions'] = peak_predictions_best_model
    
    indices_dnm2_positive = gmm_class_indices[index_DNM2positive]
    
    indices_dnm2_positive_and_ccp = indices_dnm2_positive[np.where(np.array(peak_predictions_best_model)==1)[0]]
    
    ccp_status_all_tracks = np.zeros(len(merged_all_valid_tracks))
    ccp_status_all_tracks[indices_dnm2_positive_and_ccp] = 1
    df['ccp_status'] = ccp_status_all_tracks
    print('saving dataframe...\n')
    # save the dataframe for subsequent notebooks
    compression_opts = dict(method='zip',
                            archive_name=path_outputs+'/dataframes/df_merged_features.csv')  

    df.to_csv(path_outputs+'/dataframes/df_merged_features.zip', index=False,
                                                              compression=compression_opts) 
    
    print('done\n')
    np.save(analysis_metasdata.item().get('path_outputs')+'/dataframes/analysis_metadata', analysis_metadata)
    
    return distance_best_fit, height_best_fit, width_best_fit, peak_predictions_best_model

def find_single_peak_events_dist_height_width(test_distance, 
                                              test_height, 
                                              test_width,
                                              filtered_amplitudes,
                                              tracks_dnm2_positive):
    """fit lifetimes of single-peaked items found with specified peak finding paremeters to target Rayleigh distribution"""
    print('checking parameter combo: distance -', 
          test_distance, 
          'height - ',
          test_height, 
          'width - ',
           test_width)
#     print('test')
#     print(len(filtered_amplitudes))
#     print(len(tracks_dnm2_positive))
    peak_detections_with_widths = {} 
    number_of_peaks_in_event_per_model = {}
    
    num_peaks = []
    
    current_param_outputs = [] # one-hot encoding of indices of tracks with a single peak (0: multiple peaks)

    num_single_peaked = 0 # initialize total number of tracks with a single peak

    for i in range(len(filtered_amplitudes)): # iterate through all filtered amplitudes
#         print(i)
#         print(len(tracks_dnm2_positive))
#         print(tracks_dnm2_positive)
        pvals_dnm2 = return_track_attributes.return_pvals_detection_no_buffer(tracks_dnm2_positive, i, 1)
        
        # measure whether there is 1 peak with the specified peak-finding parameters
        if len(signal.find_peaks(filtered_amplitudes[i], 
                                 distance=test_distance, 
                                 height=test_height,
                                 width=test_width)[0])==1 and len(np.where(np.array(pvals_dnm2)<0.01)[0])>0:

            current_param_outputs.append(1)
            num_single_peaked += 1

        else:

            current_param_outputs.append(0)
        
        num_peaks.append(len(signal.find_peaks(filtered_amplitudes[i], 
                         distance=test_distance, 
                         height=test_height,
                         width=test_width)[0]))

        
    # store the indices of single-peaked tracks along with the current search parameters
    peak_detections_with_widths['min_dist_'+str(test_distance)+'_min_height_'+str(test_height)+'_min_width_'+str(test_width)] = current_param_outputs

    number_of_peaks_in_event_per_model['min_dist_'+str(test_distance)+'_min_height_'+str(test_height)+'_min_width_'+str(test_width)] = num_peaks
    
    # indices of single-peaked tracks
    indices_true = list(np.where(np.array(current_param_outputs)==1)[0])

    # single-peaked tracks
    tracks_true = np.array(tracks_dnm2_positive)[indices_true]

    lifetimes = []

    # gather lifetimes of single-peaked tracks
    for i in range(len(tracks_true)):

        lifetimes.append(return_track_attributes.return_track_lifetime(tracks_true,i))

    # initialize a Rayleigh distribution
    ray = stats.rayleigh

    param = ray.fit(lifetimes) # fit distribution

    ks_pvals = stats.kstest(lifetimes,stats.rayleigh(*param).cdf)[1] # measure Kolmogorov-Smirnov p-value between fitted and raw lifetimes

    ad_stat = anderson_statistic(lifetimes, dist=stats.rayleigh)
    
    percentiles = np.linspace(0,100,51)
    percentile_lifetime_thresholds = np.percentile(lifetimes, percentiles)
    obs_freq, bins = np.histogram(lifetimes, percentile_lifetime_thresholds);
    cum_obs_freq = np.cumsum(obs_freq)
    
    cdf_fitted = stats.rayleigh.cdf(percentile_lifetime_thresholds, loc=param[-2], scale=param[-1]);
    
    expected_frequency = []

    for bin_ in range(len(percentiles)-1):

        expected_cdf_area = cdf_fitted[bin_+1] - cdf_fitted[bin_]
        expected_frequency.append(expected_cdf_area * len(lifetimes))

    cum_expected_frequency = np.cumsum(np.array(expected_frequency))
        
    chi_sq_stat, chi_sq_pval = stats.chisquare(cum_obs_freq, cum_expected_frequency);
    
    obs_freq, bins = np.histogram(lifetimes, percentile_lifetime_thresholds,density=True)
    pdf = stats.rayleigh.pdf(bins[1:], *param)
    sse = np.sum(np.power(obs_freq - pdf, 2.0))    
    
    return num_single_peaked, ks_pvals, chi_sq_stat,  chi_sq_pval, sse, peak_detections_with_widths, number_of_peaks_in_event_per_model # return the number of tracks with a single peak and the p-value of the goodness-of-fit

# vecetorize the function to allow for iterable inputs to search a parameter space of peak-finding constraints
vectorized_find_single_peak_events_dist_height_width = np.vectorize(find_single_peak_events_dist_height_width)