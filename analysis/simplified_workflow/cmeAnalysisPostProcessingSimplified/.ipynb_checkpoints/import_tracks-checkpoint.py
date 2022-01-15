import scipy.io as sio
from generate_index_dictionary import return_index_dictionary
index_dictionary = return_index_dictionary()

def load_tracks(track_path):
    """Return a Python structure of the tracks, sorted from decreasing lifetime, from the ProcessedTracks.mat output of cmeAnalysis, with keys designated track indices"""
    return sort_tracks_descending_lifetimes(sio.loadmat(track_path)) # sort the tracks by descending lifetime order

def sort_tracks_descending_lifetimes(tracks):
    """Sort tracks in descending lifetime order"""
    index_dictionary = return_index_dictionary()
    
    tracks = tracks['tracks'][0] # get just the data for the tracks
                                            
    tracks = zip(tracks, range(len(tracks))) # save each track with its original index
    # sort the tracks in descending lifetime, preserving each track's individual index
    tracks = sorted(tracks, key=lambda track: track[0][index_dictionary['index_lifetime_s']][0][0], reverse=True) 
    
    (tracks, indices) = zip(*tracks) 
    
    return tracks