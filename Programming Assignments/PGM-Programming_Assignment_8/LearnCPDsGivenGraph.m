function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);
D = length(size(G));    % dimension of G
B = size(G, 1);         % number of body parts

P.c = zeros(1,K);
P.clg = repmat(struct('mu_y', [], 'sigma_y', zeros(1, K), 'mu_x', [], ...
    'sigma_x', zeros(1, K), 'mu_angle', [], 'sigma_angle', zeros(1, K), ...
    'theta', []), 1, B);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:K
    P.c(k) = sum(labels(:, k)) / N;
end

% for b = 1:B
%     if D == 3, Gk = G(:, :, k); else Gk = G; end
%     if Gk(b, 1) == 0
%         % body part b has only the class variable as its parent
%         P.clg(b).mu_y = zeros(1, K);
%         P.clg(b).mu_x = zeros(1, K);
%         P.clg(b).mu_angle = zeros(1, K);
%         for k = 1:K
%             data = squeeze(dataset(logical(labels(:, k)), b, :));
%             [P.clg(b).mu_y(k), P.clg(b).sigma_y(k)] = FitGaussianParameters(data(:, 1));
%             [P.clg(b).mu_x(k), P.clg(b).sigma_x(k)] = FitGaussianParameters(data(:, 2));
%             [P.clg(b).mu_angle(k), P.clg(b).sigma_angle(k)] = FitGaussianParameters(data(:, 3));
%         end
%     else
%         % b has another body part variable as parent
%         P.clg(b).theta = zeros(K, 12);
%         for k = 1:K
%             p = Gk(b, 2);
%             data = squeeze(dataset(logical(labels(:, k)), b, :));
%             pData = squeeze(dataset(logical(labels(:, k)), p, :));
%             [thetaY, P.clg(b).sigma_y(k)] = FitLinearGaussianParameters(data(:, 1), pData);
%             [thetaX, P.clg(b).sigma_x(k)] = FitLinearGaussianParameters(data(:, 2), pData);
%             [thetaAngle, P.clg(b).sigma_angle(k)] = FitLinearGaussianParameters(data(:, 3), pData);
%             P.clg(b).theta(k, 1:4) = thetaY([end 1:end-1]);
%             P.clg(b).theta(k, 5:8) = thetaX([end 1:end-1]);
%             P.clg(b).theta(k, 9:12) = thetaAngle([end 1:end-1]);
%         end
%     end
% end

for k = 1:K
    for b = 1:B
        if D == 3, Gk = G(:, :, k); else Gk = G; end
        if Gk(b, 1) == 0
            % body part b has only the class variable as its parent
            if k == 1
                P.clg(b).mu_y = zeros(1, K);
                P.clg(b).mu_x = zeros(1, K);
                P.clg(b).mu_angle = zeros(1, K);
            end
            data = squeeze(dataset(logical(labels(:, k)), b, :));
            [P.clg(b).mu_y(k), P.clg(b).sigma_y(k)] = FitGaussianParameters(data(:, 1));
            [P.clg(b).mu_x(k), P.clg(b).sigma_x(k)] = FitGaussianParameters(data(:, 2));
            [P.clg(b).mu_angle(k), P.clg(b).sigma_angle(k)] = FitGaussianParameters(data(:, 3));
        else
            % b has another body part variable as parent
            if k == 1
                P.clg(b).theta = zeros(K, 12);
            end
            p = Gk(b, 2);
            data = squeeze(dataset(logical(labels(:, k)), b, :));
            pData = squeeze(dataset(logical(labels(:, k)), p, :));
            [thetaY, P.clg(b).sigma_y(k)] = FitLinearGaussianParameters(data(:, 1), pData);
            [thetaX, P.clg(b).sigma_x(k)] = FitLinearGaussianParameters(data(:, 2), pData);
            [thetaAngle, P.clg(b).sigma_angle(k)] = FitLinearGaussianParameters(data(:, 3), pData);
            P.clg(b).theta(k, 1:4) = thetaY([end 1:end-1]);
            P.clg(b).theta(k, 5:8) = thetaX([end 1:end-1]);
            P.clg(b).theta(k, 9:12) = thetaAngle([end 1:end-1]);
        end
    end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);
fprintf('log likelihood: %f\n', loglikelihood);
