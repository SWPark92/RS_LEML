function outLE = Compute_LE_Metric(Set1,Set2)

simFlag = false;
if (nargin < 2)
    Set2 = Set1;
    simFlag = true;
end

l1 = size(Set1,3);
l2 = size(Set2,3);
outLE = zeros(l2,l1);
if (simFlag)
    for tmpC1 = 1:l1
        logm_X = Set1(:,:,tmpC1);
        for tmpC2 = tmpC1+1:l2
            logm_Y = Set2(:,:,tmpC2);
            outLE(tmpC2,tmpC1) = norm( logm_X - logm_Y, 'fro' );
            if  (outLE(tmpC2,tmpC1) < 1e-10)
                outLE(tmpC2,tmpC1) = 0.0;
            end
            outLE(tmpC1,tmpC2) = outLE(tmpC2,tmpC1);
        end
    end
    
else
    for tmpC1 = 1:l1
        logm_X = Set1(:,:,tmpC1);
        for tmpC2 = 1:l2
            logm_Y = Set2(:,:,tmpC2);
            outLE(tmpC2,tmpC1) = norm( logm_X - logm_Y, 'fro' );
            if  (outLE(tmpC2,tmpC1) < 1e-10)
                outLE(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end


end