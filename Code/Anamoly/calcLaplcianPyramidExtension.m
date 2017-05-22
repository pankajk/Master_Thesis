function [fy, l, dstsSqrX, dstsSqrXY, indsXX, indsXY, sigma0] = calcLaplcianPyramidExtension(configParams, X, f, Y, nnData, dstsSqrX, dstsSqrXY, indsXX, indsXY)
% Out-of-sample Laplacian pyramid extension following 
% "Heterogeneous datasets representation and learning using diffusion maps
% and laplacian pyramids", Rabin and Coifman
% Here X is sample set, Y is new samples to extend to

dParams.sigma0 = 25;        % initial coarse sigma
dParams.maxIters = 15;      % maximum number of iterations
dParams.errThresh = 0.005;  % error threshold 
dParams.verbose = false;
if exist('configParams','var') && ~isempty(configParams)
    configParams = setParams(dParams, configParams);
else
    configParams = dParams;
end

sigma0 = configParams.sigma0;

% preparing affinity for sample set X and affinity matrix between 
% sample set X and new samples set Y
if ~exist('dstsSqrX','var') || isempty(dstsSqrX)
    % TODO: this block of code can be replaced with simple call of function
    % GETNEARESTNEIGHBORS
    if configParams.verbose
        h = waitbar(0, 'calcLaplcianPyramidExtension: calculating affinity matrix');
        percentDone = max(round(size(X, 2)/100),1);
    end
    
    X = X';
    % Number of points
    M = size(X,1);

    if ~exist('nnData','var') || isempty(nnData) || ~isfield(nnData,'NNIdxs') || ~isfield(nnData,'NNDist')
        % nn search
        [nnData.nnDist,nnData.nnInds] = pdist2(X,X,'euclidean','Smallest',configParams.kNN);
        nnData.nnDist = nnData.nnDist';
        nnData.nnInds = nnData.nnInds';
    end
                
    if isempty(nnData) || ~isfield(nnData,'dstsX') || ~isfield(nnData,'indsXX')
        % Using sparse matrix for affinity matrix
        numNonZeros = configParams.kNN * M;
        % Using sparse matrix for affinity matrix
        ind = 1;
        rowInds = zeros(1, numNonZeros);
        colInds = zeros(1, numNonZeros);
        vals = zeros(1,numNonZeros);
        for i = 1:M
            % calc the sparse row and column indices
            rowInds(ind : ind + configParams.kNN - 1) = i;
            colInds(ind : ind + configParams.kNN - 1) = nnData.nnInds(i,:);
            vals(ind : ind + configParams.kNN - 1)    = nnData.nnDist(i,:);
            ind = ind + configParams.kNN;
            if configParams.verbose  && mod(i,percentDone) == 0
                waitbar(i / M, h);
            end
        end
        dstsX = sparse(rowInds, colInds, vals, M, M);
        indsXX = sub2ind([M, M], rowInds, colInds);
        dstsSqrX = dstsX.^2;
        
    else
        dstsX = nnData.dstsX;
        indsXX = nnData.indsXX;
        dstsSqrX = dstsX.^2;
    end
    if configParams.verbose
        close(h);
    end
end

if ~exist('dstsSqrXY','var') || isempty(dstsSqrXY)
    Y = Y';
    [dstsXY, indsXY] = getNearestNeighbors(X, Y, configParams);
    sigma0 = 5*median(dstsX(indsXX));
    dstsSqrXY = dstsXY.^2;
end
    
%% Laplacian extension following the paper equations
[S0, s0y] = calcSl(dstsSqrX,dstsSqrXY, sigma0, 0, f, indsXX, indsXY);
fEst = S0;
fy = s0y;
estErr = inf;
nIters = 0;
l = 1;
while estErr > configParams.errThresh && nIters < configParams.maxIters
    dl = f - fEst;
    [Sl, sly] = calcSl(dstsSqrX, dstsSqrXY, sigma0, l, dl, indsXX, indsXY);
    fEst = fEst + Sl;
    fy = fy + sly;
    estErr = norm(fEst - f);
    nIters = nIters + 1;
    l = l + 1;
end

return;

function [Sl, sly] = calcSl(dstsSqr, dstsSqrXY, sigma0, l, dl, indsXX, indsXY)
[lIdxsI, lIdxsJ] = ind2sub(size(dstsSqr), indsXX);
lEntries = exp(-dstsSqr(indsXX)/(sigma0/(2^l)));
Wl =  sparse(lIdxsI,lIdxsJ,lEntries,size(dstsSqr,1),size(dstsSqr,2));
Wl = (Wl + Wl')/2;

one_over_Ql = spdiags(( 1./(sum(Wl,2)+eps )),0,size(Wl,1),size(Wl,2));
Sl = (one_over_Ql*Wl)*dl; % Kl*dl
[lIdxsI, lIdxsJ] = ind2sub(size(dstsSqrXY), indsXY);
lEntries = exp(-dstsSqrXY(indsXY)/(sigma0/(2^l)));
wly =  sparse(lIdxsI,lIdxsJ,lEntries,size(dstsSqrXY,1),size(dstsSqrXY,2));
d = sum(wly) + eps;
one_over_ql = spdiags((1./d)',0,size(wly,2),size(wly,2));

sly = (one_over_ql*wly')*dl;
return;
