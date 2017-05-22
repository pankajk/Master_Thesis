function detectionVars = calcDetectVars(im, topleftOrigin, numPatchesinDim)
% calc variables in advance to save time in calculating the similarity
% measure
% Gal Mishne

indsIm = reshape(1:size(im,1)*size(im,2),size(im,1),size(im,2));

if ~exist('numPatchesinDim','var')
    numPatchesinDim  = 5;
end

topleftOrigin = fliplr(topleftOrigin);
topMost    = topleftOrigin(1,:);
bottomMost = topleftOrigin(end,:);
idxPatches = sub2ind(size(im), topleftOrigin(:,1), topleftOrigin(:,2));

numPatchesinDimMat = repmat([numPatchesinDim numPatchesinDim],size(topleftOrigin,1),1);
upLeft      = topleftOrigin - numPatchesinDimMat;
bottomRight = topleftOrigin + numPatchesinDimMat;

addToBottomRight = max(0, 1-upLeft);
subFromUpleft(:,1) = max(0, bottomRight(:,1)-bottomMost(1));
subFromUpleft(:,2) = max(0, bottomRight(:,2)-bottomMost(2));

upLeft(upLeft(:,1) < topMost(1),   1) = topMost(1);
upLeft(upLeft(:,1) > bottomMost(1),1) = bottomMost(1);
upLeft(upLeft(:,2) < topMost(2),   2) = topMost(2);
upLeft(upLeft(:,2) > bottomMost(2),2) = bottomMost(2);

bottomRight(bottomRight(:,1) < topMost(1),   1) = topMost(1);
bottomRight(bottomRight(:,1) > bottomMost(1),1) = bottomMost(1);
bottomRight(bottomRight(:,2) < topMost(2),   2) = topMost(2);
bottomRight(bottomRight(:,2) > bottomMost(2),2) = bottomMost(2);

upLeft = upLeft - subFromUpleft;
bottomRight = bottomRight + addToBottomRight;

detectionVars.idxPatches    = idxPatches;
detectionVars.topleftOrigin = topleftOrigin;
detectionVars.upLeft        = upLeft;
detectionVars.bottomRight   = bottomRight;
detectionVars.indsIm        = indsIm;
detectionVars.sigma         = 5e-4;
detectionVars.useMask       = true;
detectionVars.numPatchesinDim = numPatchesinDim;
detectionVars.thresh        = 0.5;%round((numPatchesinDim*2+1)^2/2);

if detectionVars.useMask
    detectionVars.maskDim = max(2,ceil(0.2 * numPatchesinDim));
    detectionVars.thresh = 0.5;%round( ((numPatchesinDim*2+1)^2-(detectionVars.maskDim^2+1)^2) /2);
end
