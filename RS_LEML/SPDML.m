% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

clear;
clc;

addpath('local_manopt')
load('toy_data');

Metric_Flag = 1; %1:AIRM, 2:Stein
graph_kw = 100;
graph_kb = 0;
newDim = 10;

%initializing training structure
trnStruct.num_iter = 300;
trnStruct.X = covD_Struct.trn_X;
trnStruct.y = covD_Struct.trn_y;
trnStruct.n = size(covD_Struct.trn_X,1);
trnStruct.nClasses = max(covD_Struct.trn_y);
trnStruct.r = newDim;
trnStruct.Metric_Flag = Metric_Flag;
 
%Generating graph
nPoints = length(trnStruct.y);
trnStruct.G = generate_Graphs(trnStruct.X,trnStruct.y,graph_kw,graph_kb,Metric_Flag);

%- different ways of initializing, the first 10 features are genuine so
%- the first initialization is the lucky guess, the second one is a random
%- attempt and the last one is the worst possible initialization.

% U = orth(rand(trnStruct.n,trnStruct.r));
U = eye(trnStruct.n,trnStruct.r);
% U = [zeros(trnStruct.n-trnStruct.r,trnStruct.r);eye(trnStruct.r)];

% Create the problem structure.
manifold = grassmannfactory(covD_Struct.n,covD_Struct.r);
problem.M = manifold;

% conjugate gradient on Grassmann
problem.costgrad = @(U) supervised_WB_CostGrad_(U,trnStruct);
U  = conjugategradient(problem,U,struct('maxiter', trnStruct.num_iter));

TL_trnX = zeros(newDim,newDim,length(covD_Struct.trn_y));
for tmpC1 = 1:nPoints
    TL_trnX(:,:,tmpC1) = U'*covD_Struct.trn_X(:,:,tmpC1)*U;
end
TL_tstX = zeros(newDim,newDim,length(covD_Struct.tst_y));
for tmpC1 = 1:length(covD_Struct.tst_y)
    TL_tstX(:,:,tmpC1) = U'*covD_Struct.tst_X(:,:,tmpC1)*U;
end

if (Metric_Flag == 1)
    %AIRM
    pair_dist = Compute_AIRM_Metric(TL_tstX,TL_trnX);
elseif (Metric_Flag == 2)
    %Stein
    pair_dist = Compute_Stein_Metric(TL_tstX,TL_trnX);
else
    error('the metric is not defined');
end

[~,minIDX] = min(pair_dist);
y_hat = covD_Struct.trn_y(minIDX);
test_acc = sum(covD_Struct.tst_y == y_hat)/length(covD_Struct.tst_y) * 100;

fprintf('\n-----------------------------------------\n')
fprintf('Test Accuracy = %.2f%%.\n',test_acc);
fprintf('-----------------------------------------\n')
