# Jin_Shirazinejad_et_al_branched_actin_manuscript
code for "Asymmetric actin force production at stalled clathrin-mediated endocytosis sites", submitted

bioRxiv link: https://www.biorxiv.org/content/10.1101/2021.07.16.452693v1

The live-cell imaging data (TIRF), tracked events, plots used for analysis, and arrays used throughout our analysis can be found at: https://tinyurl.com/z29zy93j

The repo for the code development can be found ony my personal Github page: https://github.com/cynashirazinejad/track_processing

The principle routines that these notebooks perform are:
1) visualizing the dynamics of tracked events out of cmeAnalysis**
2) clustering these tracked events into groups of similarly-behaved events
3) visualizing the results of clustering to understand how clusters are similar and different
4) identifying events with characteristic peaks of protein recruitment
5) using a trained clustering model to make predictions about the identity of new data
6) comparing the dynamics of different experimental groups 
7) linking tracked events from separate tracking experiments
8) visualizing the results of multi-channel protein dynamics

** this step can be generalized to any tracking scheme with an output consisting of fitted intensities, positions, and statistical tests of detection confidences
