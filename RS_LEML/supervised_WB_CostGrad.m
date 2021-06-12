% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function [outCost,outGrad] = supervised_WB_CostGrad(U,covD_Struct)
outCost = 0;
dF = zeros(size(U));

nPoints = length(covD_Struct.y);
% U : n(n-1)/2 -> k(k-1)/2
target_dim = covD_Struct.r * (covD_Struct.r + 1) / 2;
for tmpC1 = 1:nPoints
    tmpMat =  covD_Struct.logm_X_upper(tmpC1, :) * U;
    tmpMat = tril(repmat(tmpMat(:), 1 ,target_dim));
    tmpMat = triu(tmpMat.',1) + tril(tmpMat);
    log_UX(:,:,tmpC1) = tmpMat;
end
outGrad = 0;

for i = 1:nPoints
    X_i = covD_Struct.X(:,:,i);
    parfor j = 1:nPoints
        if (covD_Struct.G(i,j) == 0)
            continue;
        end  
        X_j = covD_Struct.X(:,:,j);
        outCost = outCost + covD_Struct.G(i,j)*Compute_LE_Metric(log_UX(:,:,i) , log_UX(:,:,j));
        b_c = (covD_Struct.logm_X_upper(i, :) - covD_Struct.logm_X_upper(j, :));
        outGrad = outGrad + covD_Struct.G(i,j)* (b_c' * b_c * U);        
    end
end

end

