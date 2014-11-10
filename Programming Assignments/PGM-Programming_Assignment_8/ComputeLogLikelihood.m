function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1);    % number of examples
K = length(P.c);        % number of classes
D = length(size(G));    % dimension of G
B = size(G, 1);         % number of body parts

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N
    logClassLikelihood = log(P.c);  % log of class prior probabilities
    for k = 1:K
        if D == 3, Gk = G(:, :, k); else Gk = G; end
        for b = 1:B
            if Gk(b, 1) == 0
                % only class variable as parent
                logPartProb = lognormpdf(dataset(i, b, 1), P.clg(b).mu_y(k), P.clg(b).sigma_y(k)) + ...
                    lognormpdf(dataset(i, b, 2), P.clg(b).mu_x(k), P.clg(b).sigma_x(k)) + ...
                    lognormpdf(dataset(i, b, 3), P.clg(b).mu_angle(k), P.clg(b).sigma_angle(k));
            else
                % body part b has another parent, besides class variable
                p = Gk(b, 2);   % parent body part
                pval = [1; dataset(i, p, 1); dataset(i, p, 2); dataset(i, p, 3)];
                logPartProb = lognormpdf(dataset(i, b, 1), dot(pval, P.clg(b).theta(k, 1:4)), P.clg(b).sigma_y(k)) + ...
                    lognormpdf(dataset(i, b, 2), dot(pval, P.clg(b).theta(k, 5:8)), P.clg(b).sigma_x(k)) + ...
                    lognormpdf(dataset(i, b, 3), dot(pval, P.clg(b).theta(k, 9:end)), P.clg(b).sigma_angle(k));                
            end
            logClassLikelihood(k) = logClassLikelihood(k) + logPartProb;
        end
    end
    loglikelihood = loglikelihood + log(sum(exp(logClassLikelihood)));
end

end
