# Cyna Shirazinejad, Drubin/Barnes Lab, last modified 9/23/20

def merge_experiments(merged_tracks_input_list, merged_selected_IDs_input_list):
    """
    Merges multiple tracking experiments into one while preserving fed-in order.
    
    Args:
        merged_tracks_input_list (list of tracks): individual Python-converted ProcessedTracks.mat
        merged_selected_IDs_input_list (list of lists): lists of indices to keep from each corresponding track object
        
    Returns:
        output_tuple_tracks (tuple): merged tracks
    """
    
    if len(merged_tracks_input_list) == len(merged_selected_IDs_input_list):
        pass
    else:
        raise Exception('the length of "merged_tracks_input_list" must be equal to the length of "merged_selected_IDs_input_list"')

    output_tuple_tracks = []
    
    for i in range(len(merged_tracks_input_list)):
        
        for track_id in merged_selected_IDs_input_list[i]:
            
            output_tuple_tracks.append(merged_tracks_input_list[i][track_id])

    return tuple(output_tuple_tracks)

