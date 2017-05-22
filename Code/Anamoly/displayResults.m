function [diffusionCoordIm] = ...
    displayResults(im, configParams, reconstructVars, diffusion_mapX, dmInds, idxPatches, figId, results)
% various plots and images for display and debugging

detection    = results.detection;
[nrows, ncols, nlayers] = size(im);
patchSize = configParams.patchDim;

% color image according to first three coordiantes of diffusion map
diffusionCoordIm = getDiffusionCoordIm(im, idxPatches, diffusion_mapX);
if configParams.saveResults
    filename = [configParams.path_name configParams.image_name 'DMCoordIm.png'];
    imwrite(diffusionCoordIm, filename);
end
if configParams.plotResults
    figure;
    imshow(diffusionCoordIm);
    title('Image in Diffusion Coordinates')
end

% binary mask of samples used to construct diffusion map
dmIndsIm = false(nrows, ncols);
dmIndsIm(idxPatches(dmInds)) = true;
if configParams.saveResults
    filename = [configParams.path_name configParams.image_name 'samples.png'];
    imwrite(dmIndsIm, filename);
end

% plot diffusion map colored in RGB
M = length(idxPatches);
C = plotDiffusionMapinColor(diffusionCoordIm, idxPatches, diffusion_mapX, 1:M, figId,'Diffusion Map',10);
if configParams.saveResults
    filename = [configParams.path_name configParams.image_name '_GaussPyrDM.png'];
    saveas(gcf,filename) ;
end

% plot results
if configParams.plotResults
    if ~isempty(figId)
        h = figure(figId);
    else
        h = figure;
    end
    
    subplot(221);
    imshow(im)
    title('Image');
    
    subplot(222);
    imshow(diffusionCoordIm);
    title('Image in Diffusion Coordinates');
    
    subplot(223);
    imshow(dmIndsIm);
    title('Diffusion Map Sample Set');
    
    subplot(224)
    detIm = zeros(nrows - patchSize + 1, ncols - patchSize + 1);
    detIm(1:end) = detection;
    imshow(detIm,[0 1])
    colorbar EastOutside
    title('Raw detection score');
    
    if configParams.saveResults
        filename = [configParams.path_name configParams.image_name '_GaussPyr.png'];
        saveas(gcf,filename) ;
    end
end

return
