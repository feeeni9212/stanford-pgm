function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

M = size(U,1);
N = size(U,2);

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

% construct A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = ones(N+1, N+1);
A(1, 1:end-1) = mean(U);
A(2:end, end) = reshape(mean(U), [], 1);

for i = 1:N
    Ui = repmat(U(:, i), 1, N);
    A(i+1, 1:end-1) = mean(U.*Ui);
end

% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]

% construct B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = reshape([mean(X), mean(repmat(X, 1, N).*U)], [], 1);

% solve A*Beta = B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Beta = A \ B;

% then compute sigma according to eq. (11) in PA description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = Beta(1:end-1);
sigma = sqrt(cov(X, 1) - sum(sum((b*b').*cov(U, 1))));
