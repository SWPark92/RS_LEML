%{
This implementation is the modified version of the original code of the following work.
'M.Harandi, M. Salzmann and R. Hartley, "From manifold to manifold: geometry-aware dimensionality reduction for SPD matrices'
%}

clear;
clc;

path = 'toy_data';
load(path); % Load dataset.
%load('logm_X'); % Load pre-processed training data.

graph_kw = 100;
graph_kb = 0;
newDim = 10;

for i = 1:125 % Pre-process the training data.
    trn_logm_X(:,:,i) = logm(covD_Struct.trn_X(:,:,i));
    tst_logm_X(:,:,i) = logm(covD_Struct.tst_X(:,:,i));
end
save('logm_X.mat' , 'trn_logm_X', 'tst_logm_X');
load('logm_X');

%initializing training structure
trnStruct.num_iter = 300;
trnStruct.learningRate = 1E-4; % 
trnStruct.RSType = 'Grasmann';
trnStruct.tst_logm_X = tst_logm_X;
trnStruct.trn_logm_X = trn_logm_X;
trnStruct.logm_X_upper = get_upper_triangluar(trn_logm_X);
trnStruct.logm_X_upper_ = get_upper_triangluar(tst_logm_X);
trnStruct.X = covD_Struct.trn_X;
trnStruct.y = covD_Struct.trn_y;
trnStruct.n = size(covD_Struct.trn_X,1);
trnStruct.nClasses = max(covD_Struct.trn_y);
trnStruct.r = newDim;

% Generating graph for defining objective functions for metric learning.
nPoints = length(trnStruct.y);
trnStruct.G = generate_Graphs(trnStruct.X,trnStruct.y,graph_kw,graph_kb, 3);
RS_Dim = trnStruct.r * (trnStruct.r + 1)/2;

% The initial point 'X' is defined on the Riemannian sub-manifold of 
% ambient manifold M = R^( (n+1)n/2, (r+1)r/2). This initial point is posibblly far from the optimal point.
X = eye( trnStruct.n * (trnStruct.n + 1)/2, RS_Dim) / (trnStruct.r * (trnStruct.r + 1)/2);

% Define the function producing losses and its corresponding Euclidean gradient on Lie-group.
costgrad = @(X) supervised_WB_CostGrad(X,trnStruct);

for j = 1 : trnStruct.num_iter
    [outCost,outEGrad] = costgrad(X);
    
    % While the proposed method utilizes 'Riemannian submanifold' of
    % ambient M = R^( (n+1)n/2, (r+1)r/2), the Euclidean metric should be
    % projected onto the Lie-subalgebra of M.
    % For the detailed implementations, please refer to 'manopt.org'.
    
    switch (trnStruct.RSType)
        case 'Euclidean'
            outRGrad = outEGrad;
        case 'Grasmann'
            outProj = multiprod(multitransp(X), outEGrad);
            outRGrad = outEGrad - multiprod(X, outProj);
        case 'Stiefel'
            outProj = multiprod(multitransp(X), outEGrad);
            symoutProj = multisym(outProj);
            outRGrad = multiprod(X, symoutProj);
        otherwise
            fprintf('Unimplemented...\n')
    end
    X = X - trnStruct.learningRate * outRGrad;
    fprintf('[%d] Cost = %f, GradientNorm = %f\n', j, outCost,  norm(outRGrad));
end

for tmpC1 = 1:nPoints
    tmpMat = trnStruct.logm_X_upper(tmpC1, :) * X;
    tmpMat = tril(repmat(tmpMat(:), 1 , RS_Dim));
    tmpMat = triu(tmpMat.',1) + tril(tmpMat);
    TL_trnX(:,:,tmpC1) = tmpMat;
end

parfor tmpC1 = 1:length(covD_Struct.tst_y)
    tmpMat = trnStruct.logm_X_upper_(tmpC1, :) * X;
    tmpMat = tril(repmat(tmpMat(:), 1 ,RS_Dim));
    tmpMat = triu(tmpMat.',1) + tril(tmpMat);
    TL_tstX(:,:,tmpC1) = tmpMat;    
end

pair_dist= Compute_LE_Metric(TL_tstX,TL_trnX);

[~,minIDX] = min(pair_dist);
y_hat = covD_Struct.trn_y(minIDX);
test_accuracy = sum(covD_Struct.tst_y == y_hat)/length(covD_Struct.tst_y) * 100;

fprintf('\n-----------------------------------------\n')
fprintf('Test Accuracy -> %.3f%%.\n',test_accuracy);
fprintf('-----------------------------------------\n')