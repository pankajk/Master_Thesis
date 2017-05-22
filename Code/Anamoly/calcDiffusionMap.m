function [diffusion_map, Lambda, Psi, nnData] = calcDiffusionMap(X,configParams)
% calculate diffusion map of data matrix X, size N by M 
% N - length of feature vector, M - number of vectors
% affinity matrix is calculated for kNN nearest neighbors, resulting in
% sparse matrix. This saves on runtime.
% the scale for the affinity matrix can be set using auto-tuning
%
% Gal Mishne

dParams.kNN       = 4; % number of nearest neighbors to consider in affinity matrix
dParams.self_tune = 0; % if true sets local scale for each sample using auto-tuning
dParams.verbose   = true;
% set default values for parameters which are missing 
configParams = setParams(dParams, configParams);

[N, M] = size(X);
if configParams.verbose
    percentDone = max(round(M/100),1);
    h = waitbar(0,'Calcuating Affinity Matrix');
end

%% affinity matrix
X = X';
[nnData.nnDist, nnData.nnInds] = pdist2(X,X,'euclidean','Smallest',configParams.kNN);
nnData.nnDist = nnData.nnDist';
nnData.nnInds = nnData.nnInds';

% Total number of entries in the W matrix
numNonZeros = configParams.kNN * M;
% Using sparse matrix for affinity matrix
ind = 1;
% initialize
rowInds      = zeros(1, numNonZeros);
colInds      = zeros(1, numNonZeros);
vals         = zeros(1,numNonZeros);
autotuneVals = zeros(1,numNonZeros);
if configParams.self_tune
    nnAutotune = min(configParams.self_tune,size(nnData.nnDist,2));
    sigmaKvec  = (nnData.nnDist(:,nnAutotune));
end
for i = 1:M
    % calc the sparse row and column indices
    rowInds(ind : ind + configParams.kNN - 1) = i;
    colInds(ind : ind + configParams.kNN - 1) = nnData.nnInds(i,:);
    vals(ind : ind + configParams.kNN - 1)    = nnData.nnDist(i,:);
    if configParams.self_tune
        autotuneVals(ind : ind + configParams.kNN - 1) = sigmaKvec(i) * sigmaKvec(nnData.nnInds(i,:));
    end
    ind = ind + configParams.kNN;
    if configParams.verbose  && mod(i,percentDone) == 0
        waitbar(i / M, h);
    end
end
% sparse affinity matrix of dataset X
nnData.dstsX  = sparse(rowInds, colInds, vals, M, M);
nnData.indsXX = sub2ind([M, M], rowInds, colInds);
autotuneMat   = sparse(rowInds, colInds, autotuneVals, M, M);
clear rowInds colInds vals autotuneVals

if configParams.verbose
    close(h)
end

% setting local scale for each sample, 
% following "Self-tuning Clustering", Zelink-Manor and Perona
if configParams.self_tune
    K                 = nnData.dstsX.^2;
    K( nnData.indsXX) = K(nnData.indsXX) ./ (autotuneMat( nnData.indsXX) + eps);
    K(nnData.indsXX)  = exp(-K( nnData.indsXX));
    K                 = spdiags(ones(size(nnData.dstsX,1),1),0,K);
else
    % if not auto-tuning, use the std of the distances as scale
    sig = std(nnData.dstsX(:)); 
    K   = exp(-nnData.dstsX.^2/(sig^2));
end
% symmetriz the matrix
K = (K + K')/2;
D = sum(K,2) + eps;

%% calculate eigen decomposition
one_over_D_sqrt = spdiags(sqrt(1./D),0,size(K,1),size(K,2));
% using symmetric matrix for calculation
Ms = one_over_D_sqrt * K * one_over_D_sqrt;
if ~issparse(Ms) && sum(Ms(:)==0)/numel(Ms)>0.75
    Ms(Ms<1e-10) = 0;
    Ms = sparse(Ms);
end

if configParams.verbose
    disp('Calculating Eigen values and vectors');
end

options.disp = 0;
options.isreal = true;
options.issym = true;
% calculating first configParams.maxInd eigenvalues and vectors
[v,lambda] = eigs(Ms,configParams.maxInd,'lm',options);
Lambda     = diag(lambda);
[Lambda,I] = sort(Lambda,'descend'); % eigs doesn't necessarily return the values sorted
v          = v(:,I);
Psi        = one_over_D_sqrt * v;
Psi        = Psi./ repmat(sqrt(sum(Psi.^2)),size(Psi,1),1);
inds       = 2:length(Lambda); % disregarding first trivial eigenvalue
clear one_over_D_sqrt

% diffusion map for t=1
diffusion_map = (Psi(:,inds).*repmat(Lambda(inds)',size(Psi,1),1))';

% plotting results
if configParams.displayResults
    figure;
    plot(Lambda);
    title('\lambda');
    ylim([0 1])
    
    figure;
    scatter(1:size(Psi,1),diffusion_map(1,:))
    hold all
    scatter(1:size(Psi,1),diffusion_map(2,:))
    scatter(1:size(Psi,1),diffusion_map(3,:))
    title('Diffusion Map Coordinates');
end

if configParams.displayResults
    figure(101);
    scatter3(diffusion_map(1,:),diffusion_map(2,:),diffusion_map(3,:))
    title('Diffusion Map');
end
