function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
B = size(dataset, 2);
K = size(labels, 2);
D = length(size(G));    % dimension of G
pred = zeros(N, K);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N
    logClassLikelihood = log(P.c);  % log of class prior probabilities
    for k = 1:K
        for b = 1:B
            if D == 3, Gk = G(:, :, k); else Gk = G; end
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
    [~, indx] = max(logClassLikelihood);
    pred(i, indx) = 1;
end

accuracy = sum(labels(:, 1) == pred(:, 1)) / N;
fprintf('Accuracy: %.2f\n', accuracy);
