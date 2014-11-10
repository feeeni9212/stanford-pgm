function aclose = allClose(A, B)

if ~all(size(A) == size(B))
    error('Dimensions of the two arrays do not match');
end

epsilon = 1e-8;
D = abs(A - B);
aclose = max(D(:)) < epsilon;
