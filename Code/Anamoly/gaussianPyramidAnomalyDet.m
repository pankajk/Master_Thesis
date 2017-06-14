function gaussianPyramidAnomalyDet(im, path_name, image_name, saveRes, testFullSS, testSS)
% this function is the main function for multiscale anomaly detection
% calculates the Gaussian Pyramid 
% sets the different parameters for detection at each level
% upscale the suspicious pixel to scale of next level
% outputs final detection score

close all
if ~exist('im','var')
    im  = im2double(imread('./Master_Thesis/Code/Anamoly/data/sheep.jpg'));
end
if ~exist('path_name','var')
    path_name  = './Master_Thesis/Code/Anamoly/output_sheep/';
end
if ~exist('image_name','var')
    image_name  = 'sheep';
end
if ~exist('saveRes','var')
    saveRes  = true;
end
if ~exist('testFullSS','var')
    testFullSS  = false;
end
if ~exist('testSS','var')
    testSS  = false;
end

im = im2double(im);
% simple preprocessing
maxval = max(im(:));
minval = min(im(:));
im     = (im - minval) / (maxval - minval);
filename = [path_name image_name '.jpg'];
imwrite(im, filename);

%% initialize configuration parameters
if ~testSS
    maxPyrLevel = 3; % number of levels in pyramid
    dmSizeRatio = [0.1 1/3 0.5]; % sampling ratio for diffusion map at each level of the pyramid
else
    maxPyrLevel = 1; % number of levels in pyramid
    dmSizeRatio = 0.2; % sampling ratio for diffusion map at each level of the pyramid   
end
configParams.saveResults     = saveRes; % save output figures and images to path_name directory
configParams.plotResults     = false; % plot intermiediate results
configParams.verbose         = false; % display waitbars and messages
configParams.kNN             = 16;   % number of nearest neighbors used in calculation of affinity matrix
configParams.self_tune       = 7;    % index of neighbor used for self-tuning
configParams.fullPatchDim    = 8;    % size of patch at full-size level of the pyramid
configParams.testFullSS      = testFullSS; % single scale version, no sampling - all pixels are used  
configParams.useSaliency     = true;
dmInds = []; % initiliaze empty array of diffusion map indices

if configParams.saveResults
    cmap = colormap(jet(256));
    colormap('default');
end

%%
if configParams.testFullSS 
    maxPyrLevel = 1; % number of levels in pyramid
    dmSizeRatio = 1; % sampling ratio for diffusion map at each level of the pyramid
end
%% build gaussian pyramid
pyr(1).im = im;
for pyrLevel = 2:maxPyrLevel
    pyr(pyrLevel).im = impyramid(pyr(pyrLevel-1).im, 'reduce');
end

%%
% go over pyramid levels, from coarse to fine
for pyrLevel = maxPyrLevel:-1:1

    if configParams.plotResults     
        figure;imshow(pyr(pyrLevel).im); 
        title(['Image, pyramid level = ' num2str(pyrLevel)] )
    end
    % set parameters specific to pyramid level
    configParams.patchDim        = floor(configParams.fullPatchDim/pyrLevel); % patch is of sie patchDim^2
    configParams.maxInd          = min(configParams.patchDim^2, 7); % maximum number of eigenvalues to consider for DM
    configParams.numPatchesinDim = round(20/pyrLevel); % size of neighborhood to look for anomaly
    configParams.dmSizeRatio     = dmSizeRatio(pyrLevel); 
    configParams.stdmult = 20;
        
    image_name_pyr = [image_name '_pyr_' num2str(pyrLevel)];
    
    inputs.im                   = pyr(pyrLevel).im;
    inputs.image_name           = image_name_pyr;
    inputs.path_name            = path_name;
    inputs.dmInds               = dmInds;

    [dmInds, pyr(pyrLevel).detIm ] = ...
        diffusionAnomalyDetection(inputs, configParams);

    if configParams.saveResults
        filename = [path_name image_name_pyr 'detect.png'];
        temp     = pyr(pyrLevel).detIm;
        temp(isnan(temp)) = 0;
        temp     = im2uint8(temp);
        imwrite(temp, cmap, filename);
    end

    if pyrLevel > 1
        % upscale the suspicious sample locations to size of next pyramid level
        [r, c] = ind2sub([size(pyr(pyrLevel).im,1) size(pyr(pyrLevel).im,2)],dmInds);
        r = [r*2 - 1 ; r*2     ; r*2 - 1; r*2 ];
        c = [c*2 - 1 ; c*2 - 1 ; c*2    ; c*2 ];
        invalid_inds = r>size(pyr(pyrLevel-1).im,1) | c>size(pyr(pyrLevel-1).im,2);
        r(invalid_inds) = [];
        c(invalid_inds) = [];
        dmInds = sub2ind([size(pyr(pyrLevel-1).im,1) size(pyr(pyrLevel-1).im,2)],r, c);
    end
end

% prepare smoothed anomaly score image of final level
confidence = pyr(1).detIm;
smoothedConfidence = confidence;
smoothedConfidence(confidence < 0.3) = 0; % trim low scores, then smooth
smoothedConfidence = imfilter(smoothedConfidence, fspecial('average',3));

if configParams.saveResults
    filename = [path_name image_name '_smoothConf.png'];
    smoothedConfidence(isnan(smoothedConfidence)) = 0;
    imwrite(im2uint8(smoothedConfidence), cmap, filename);
end

if configParams.plotResults
    figure;
    subplot(121)
    imshow(confidence,[0 1]);
    title('Raw Anomaly score')
    subplot(122)
    imshow(smoothedConfidence,[]);
    colormap jet
    title('Smoothed Anomaly score')
    if configParams.saveResults
        filename = [path_name image_name '_GaussPyrConf.png'];
        saveas(gcf,filename) ;
    end
    
    figure;
    for pyrLevel = maxPyrLevel : -1 : 1 
        subplot(1,maxPyrLevel,maxPyrLevel-pyrLevel+1);
        imshow(pyr(pyrLevel).detIm,[]);
        title(['Anomaly scores, l=' num2str(pyrLevel)]);
    end
    colormap jet
    
    if configParams.saveResults
        filename = [path_name image_name '_GPyrDetIm.png'];
        saveas(gcf,filename) ;
    end
    
end

return;
