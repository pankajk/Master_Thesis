%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB code implementation of multiscale anomaly detection 
% algorithm presented in 
% Gal Mishne and Israel Cohen, "Multiscale Anomaly Detection 
% Using Diffusion Maps", IEEE Journal of Selected Topics in 
% Signal Processing, Vol. 7, Number 1, February 2013, pp. 
% 111-123.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

main program is gaussianPyramidAnomalyDet.m 
It calculates multiscale levels of the image and sets the parameters for detection at every level.

This function call diffusionAnomalyDetection which performs dimensionality reduction and anomaly detection for a single scale.
sub-functions:
    * getDiffusionMapWithExtend - below
    * calcDetectVars - sets parameters for detection, used to save on runtime by calculating necessary parameters in advance
    * anoDetection - calculates similarity measure based on the reduced dimensionality representation (diffusion map coordiantes)
    * displayResults - various plots and images for display and debugging

getDiffusionMapWithExtend calculates the diffusion map for subset and extends to entire dataset.
sub-functions:
    * calcDiffusionMap - calculates sparse affinity matrix based on which the diffusion map is calculated
    * calcLaplcianPyramidExtension - extends the diffusion map from the subset to entire dataset using Laplacian Pyramid extension. 
    (Other out-of-sample extension methods could be used such as Nystrom, Geometric Harmonics)
    Note: there is lack of consistency in the notations between this function and calling functions. 
    In calling functions Y is the subset, X is entire dataset. Here the notations are reversed.

To use the saliency based score set 
configParams.useSaliency     = true;
in the file gaussianPyramidAnomalyDet.m 

auxiliary fuctions
* setParams - useful for setting defaults for expected inputs 



