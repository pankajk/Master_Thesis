function [results] = anoDetection(detectionVars, diffusion_mapX)
% Calculates a similarity measure between a pixel and its spatial neighbors
% (from the image), using the reduced dimensionality representation of each
% pixel. If prior knowledge of the size of the anomaly exists, a mask can
% be used to ignore the close neighborhood of the pixel and consider only
% the far neighborhood (ring around the pixel)
%
% Gal Mishne

useSaliency = detectionVars.useSaliency;
topleftOrigin = detectionVars.topleftOrigin;

if useSaliency
    clear detectionVars
    [nnData.nnDist, nnData.nnInds] = pdist2(diffusion_mapX',diffusion_mapX','euclidean','Smallest',64);
    sigma = 2 * std(nnData.nnDist(64,:));
    nnData.nnDist = nnData.nnDist ./ sigma;
    nnData.nnDist = nnData.nnDist';
    nnData.nnInds = nnData.nnInds';
    rows = topleftOrigin(:,1);
    nnRows = rows(nnData.nnInds);
    cols = topleftOrigin(:,2);
    nnCols = cols(nnData.nnInds);
    max_dim = max( max(rows),max(cols) );
    pixel_dists = sqrt((repmat(nnRows(:,1),1,size(nnRows,2)-1) - nnRows(:,2:end)).^2 + ...
        (repmat(nnCols(:,1),1,size(nnCols,2)-1) - nnCols(:,2:end)).^2) ./ max_dim;
    dists = nnData.nnDist(:,2:end) ./ (1+3*pixel_dists);
    similarity = exp(-( mean(dists,2  ) ));
else
    idxPatches    = detectionVars.idxPatches;
    upLeft        = detectionVars.upLeft;
    bottomRight   = detectionVars.bottomRight;
    indsIm        = detectionVars.indsIm;
    useMask       = detectionVars.useMask;
    sigma         = detectionVars.sigma;
    if useMask
        maskDim = detectionVars.maskDim;
    end
    clear detectionVars
    
    similarity = zeros(size(upLeft,1),1);
    
    nPatches = size(upLeft,1);
    normDiffusionCoords = sum(diffusion_mapX.^2);
    
    parfor i = 1 : nPatches
        rows = upLeft(i,1):bottomRight(i,1);
        cols = upLeft(i,2):bottomRight(i,2);
        if useMask
            % remove vectors which are in the masked zone of each pixel neighborhood
            mask = true(length(rows),length(cols));
            
            masked_inds = [topleftOrigin(i,1) + [-maskDim;maskDim], topleftOrigin(i,2) + [-maskDim;maskDim]];
            masked_inds = max(masked_inds,1);
            masked_inds(:,1) = min(masked_inds(:,1),max(bottomRight(:,1)));
            masked_inds(:,2) = min(masked_inds(:,2),max(bottomRight(:,2)));
            masked_inds = masked_inds - [upLeft(i,:);upLeft(i,:)] + 1;
            mask(masked_inds(1,1):masked_inds(2,1),masked_inds(1,2):masked_inds(2,2)) = false;
            winIdx = indsIm(rows,cols).*mask;
        else
            winIdx = indsIm(rows,cols);
        end
        winIdx = winIdx(winIdx~=0);
        
        [~, winIdxPatches, ~] = intersect(idxPatches, winIdx);
        
        diffusionMapCoords = diffusion_mapX(:,winIdxPatches);
        patchCoords = diffusion_mapX(:,i);
        
        dists = normDiffusionCoords(:,winIdxPatches) - 2*patchCoords'*diffusionMapCoords + normDiffusionCoords(:,i);
        W = exp(-dists/(sigma+eps));
        W(W<0.5) = 0;
        
        similarity(i) = mean(W);
        
        if similarity(i) == 0
            W = zeros(size(W));
        else
            W = W/sum(W);
        end
    end
    
end

results.detection = 1 - similarity;

return;

