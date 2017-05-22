function [dmInds, detIm] = diffusionAnomalyDetection(inputs, configParams)
% this function perform anomaly detection at a single scale
% transforms image into patches ordered in column stack
% sets the samples to be used in construction of the diffusion map
% calculates the diffusion map and extends it to all pixels
% performs anomaly detection in the reduced dimensionality 

if ~exist('inputs','var')
    disp('error: function needs inputs struct with fields im, image_name, path_name, dmInds');
    return;
else
    im                      = inputs.im;
    configParams.image_name = inputs.image_name;
    configParams.path_name  = inputs.path_name;
    dmInds                  = inputs.dmInds;
    clear inputs
end

% for controlled random input
s = RandStream('mt19937ar');
RandStream.setGlobalStream(s);

% sets defaults for parameters which were not set by calling function
configParams = setAnomalyDetParams(configParams);

%% extract overlapping square patches from image
% feature used is patches of image, if other feature is desirable replace
% this block of code
% X - matrix of size patchdim^2*num_patches, patches of the images organized as coloumns
if size(im,3) ~= 1
    %im = im2uint8(im);
    cform = makecform('srgb2lab');
    lab_im = applycform(im,cform);
    lab_im = lab2double(lab_im);
    X=[];
    for i = 1:3
        [Xi, topleftOrigin] = im2patch(lab_im(:,:,i), configParams.patchDim);
        X=[X;Xi];
    end
else
    [X, topleftOrigin] = im2patch(im, configParams.patchDim);
end
featuresLoc        = topleftOrigin + floor(configParams.patchDim/2); % return center of each patch
idxPatches = sub2ind([size(im,1) size(im,2)], featuresLoc(:,2), featuresLoc(:,1));
M                  = length(idxPatches); % M is number of patches

%% dimensionality reduction
% calculate size of diffusion map, restrict to maximum size of 5000 samples
if isempty(configParams.dmSizeRatio)
    if M < 1000
        dmSize = M;
    else
        dmSize = max(1000,min(round(M*0.1),5000));
    end
else
    dmSize = round(M*configParams.dmSizeRatio);
end
[temp, indsInX, indsInY] = intersect(idxPatches, dmInds);
dmInds = indsInX;
if length(dmInds) < dmSize
    % add random samples
    allInds = (1:M)';
    allInds = setdiff(allInds,dmInds);
    p = randperm(length(allInds));
    allInds = allInds( p(1:(dmSize - length(dmInds) )));
    dmInds  = sort([dmInds; allInds]);
elseif length(dmInds) > dmSize
    % decrease number of samples
    dmInds = dmInds(round(linspace(1,length(dmInds),dmSize)));
end

% Y is the subset gamma, samples from the image used to construct the
% diffusion map
Y = X(:,dmInds);
% calculate diffusion map for Y and extend to all of X
diffusion_mapX = getDiffusionMapWithExtend(X, Y, configParams);

%% anomaly detection
if configParams.useSaliency
    detectionVars.topleftOrigin = topleftOrigin;
else
    detectionVars = calcDetectVars(im, topleftOrigin,configParams.numPatchesinDim);
    
    % calculate scale parameter used in similarity measure based on random
    % pairs of distances in the reduced dimensionality
    n_compare = 2*min(1000, round(M/4));
    p = randperm(M);
    p = p(1:n_compare);
    d = zeros(1,n_compare/2);
    for i = 1:n_compare/2
        d(i) = sqrt(sum(diffusion_mapX(:,p(i)) - diffusion_mapX(:,p(n_compare/2+i))).^2);
    end
    % stdmult is a parameter that affects the detection results, can be fine-tuned manually...
    detectionVars.sigma = configParams.stdmult*std(d)^2;
end
detectionVars.useSaliency = configParams.useSaliency;
% display reconstructed image
[results] = anoDetection(detectionVars, diffusion_mapX);

%%
% plot results
figId = [];
if configParams.plotResults || configParams.saveResults
    displayResults(im, configParams, detectionVars, diffusion_mapX, dmInds, idxPatches, figId, results);
end

% determine suspicious samples, using 0.05 quantile of similairty measure
temp = zeros(size(im,1),size(im,2)); % binary image of suspicious samples
thresh = quantile(results.detection(:),0.95);
temp(idxPatches) = results.detection(:)>thresh;
dmInds = find(temp);
if configParams.saveResults
    filename = [configParams.path_name configParams.image_name 'suspicious.png'];
    imwrite(temp, filename);
end

detIm = nan(size(im,1),size(im,2));
detIm(idxPatches) = results.detection(:);

return;

function configParams = setAnomalyDetParams(configParams)
dParams.patchDim        = 8;
dParams.includeMargins  = 0;
dParams.saveResults     = false;
dParams.sig             = 0.1;
dParams.kNN             = 8;
dParams.self_tune       = 4;
dParams.verbose         = false;
dParams.displayResults  = false;
dParams.maxInd          = 7;
dParams.numPatchesinDim = 10;
dParams.plotResults = false;
dParams.dmSizeRatio = [];
dParams.stdmult = 20;
dParams.useSaliency  = false;
if exist('configParams','var')
    configParams = setParams(dParams, configParams);
else
    configParams = dParams;
end

