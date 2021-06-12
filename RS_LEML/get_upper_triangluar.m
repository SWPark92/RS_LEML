function [v] = get_upper_triangluar(A)

for i=1:size(A,3)
At = A(:,:,i).';
m  = tril(true(size(At)));
v(i, :)  = At(m).';
end
end
