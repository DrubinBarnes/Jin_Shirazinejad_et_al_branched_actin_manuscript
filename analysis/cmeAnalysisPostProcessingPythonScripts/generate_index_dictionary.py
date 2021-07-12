# Cyna Shirazinejad, 10/7/20
import os
import sys
sys.path.append(os.getcwd())

def return_index_dictionary():
    """Hard-coding all indices of outputs in the array for each track produced by cmeAnalysis."""

    index_dictionary = {}

    index_dictionary['index_time_frames'] = 0
    index_dictionary['index_frames'] = 1
    index_dictionary['index_x_pos'] = 2
    index_dictionary['index_y_pos'] = 3
    index_dictionary['index_amplitude'] = 4
    index_dictionary['index_background'] = 5
    index_dictionary['index_x_pos_pstd'] = 6
    index_dictionary['index_y_pos_pstd'] = 7
    index_dictionary['index_amplitude_pstd'] = 8
    index_dictionary['index_background_pstd'] = 9
    index_dictionary['index_sigma_r'] = 10
    index_dictionary['index_SE_sigma_r'] = 11
    index_dictionary['index_pval_Ar'] = 12
    index_dictionary['index_isPSF'] = 13
    index_dictionary['index_tracksFeatIndxCG'] = 14
    index_dictionary['index_gapVect'] = 15
    index_dictionary['index_gapStatus'] = 16
    index_dictionary['index_gapIdx'] = 17
    index_dictionary['index_seqofEvens'] = 18
    index_dictionary['index_nSeg'] = 19
    index_dictionary['index_visibility'] = 20
    index_dictionary['index_lifetime_s'] = 21
    index_dictionary['index_start'] = 22
    index_dictionary['index_end'] = 23
    index_dictionary['index_startBuffer'] = 24
    index_dictionary['index_endBuffer'] = 25
    index_dictionary['index_MotionAnalysis'] = 26
    index_dictionary['index_maskA'] = 27
    index_dictionary['index_maskN'] = 28
    index_dictionary['index_RSS'] = 29
    index_dictionary['index_mask_Ar'] = 30
    index_dictionary['index_hval_Ar'] = 31
    index_dictionary['index_hval_AD'] = 32
    index_dictionary['index_catIdx'] = 33
    index_dictionary['index_isCCP'] = 34
    index_dictionary['index_significantMaster'] = 35
    index_dictionary['index_significantVsBackground'] = 36
    index_dictionary['index_significantSlave'] = 37

    return index_dictionary