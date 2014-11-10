% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);          % number of poses
K = size(InitialClassProb, 2);  % number of classes (clusters)

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

D = length(size(G));            % dimension of G
B = size(G, 1);                 % number of body parts
Gk = G;

P.c = [];
% P.clg.sigma_x = [];
% P.clg.sigma_y = [];
% P.clg.sigma_angle = [];

P.clg = repmat(struct('mu_y', [], 'sigma_y', zeros(1, K), 'mu_x', [], ...
    'sigma_x', zeros(1, K), 'mu_angle', [], 'sigma_angle', zeros(1, K), ...
    'theta', []), 1, B);

% EM algorithm
for iter=1:maxIter

    % M-STEP to estimate parameters for Gaussians
    %
    % Fill in P.c with the estimates for prior class probabilities
    % Fill in P.clg for each body part and each class
    % Make sure to choose the right parameterization based on G(i,1)
    %
    % Hint: This part should be similar to your work from PA8

    P.c = sum(ClassProb) / N;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for k = 1:K
        ClassProbK = ClassProb(:, k);
        for b = 1:B
%             if D == 3, Gk = G(:, :, k); else Gk = G; end
            if Gk(b, 1) == 0
                % body part b has only the class variable as its parent
                if k == 1
                    P.clg(b).mu_y = zeros(1, K);
                    P.clg(b).mu_x = zeros(1, K);
                    P.clg(b).mu_angle = zeros(1, K);
                end
                [P.clg(b).mu_y(k), P.clg(b).sigma_y(k)] = FitG(poseData(:, b, 1), ClassProbK);
                [P.clg(b).mu_x(k), P.clg(b).sigma_x(k)] = FitG(poseData(:, b, 2), ClassProbK);
                [P.clg(b).mu_angle(k), P.clg(b).sigma_angle(k)] = FitG(poseData(:, b, 3), ClassProbK);
            else
                % b has another body part variable as parent
                if k == 1
                    P.clg(b).theta = zeros(K, 12);
                end
                p = Gk(b, 2);
                pData = squeeze(poseData(:, p, :));
                [thetaY, P.clg(b).sigma_y(k)] = FitLG(poseData(:, b, 1), pData, ClassProbK);
                [thetaX, P.clg(b).sigma_x(k)] = FitLG(poseData(:, b, 2), pData, ClassProbK);
                [thetaAngle, P.clg(b).sigma_angle(k)] = FitLG(poseData(:, b, 3), pData, ClassProbK);
                P.clg(b).theta(k, 1:4) = thetaY([end 1:end-1]);
                P.clg(b).theta(k, 5:8) = thetaX([end 1:end-1]);
                P.clg(b).theta(k, 9:12) = thetaAngle([end 1:end-1]);
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % E-STEP to re-estimate ClassProb using the new parameters
    %
    % Update ClassProb with the new conditional class probabilities.
    % Recall that ClassProb(i,j) is the probability that example i belongs to
    % class j.
    %
    % You should compute everything in log space, and only convert to
    % probability space at the end.
    %
    % Tip: To make things faster, try to reduce the number of calls to
    % lognormpdf, and inline the function (i.e., copy the lognormpdf code
    % into this file)
    %
    % Hint: You should use the logsumexp() function here to do
    % probability normalization in log space to avoid numerical issues

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    logClassJointProb = repmat(log(P.c), N, 1);
    for i = 1:N
        for k = 1:K
            for b = 1:B
%                 if D == 3, Gk = G(:, :, k); else Gk = G; end
                if Gk(b, 1) == 0
                    % only class variable as parent
                    logPartProb = lognormpdf(poseData(i, b, 1), P.clg(b).mu_y(k), P.clg(b).sigma_y(k)) + ...
                        lognormpdf(poseData(i, b, 2), P.clg(b).mu_x(k), P.clg(b).sigma_x(k)) + ...
                        lognormpdf(poseData(i, b, 3), P.clg(b).mu_angle(k), P.clg(b).sigma_angle(k));
                else
                    % body part b has another parent, besides class variable
                    p = Gk(b, 2);   % parent body part
                    pval = [1; poseData(i, p, 1); poseData(i, p, 2); poseData(i, p, 3)];
                    logPartProb = lognormpdf(poseData(i, b, 1), dot(pval, P.clg(b).theta(k, 1:4)), P.clg(b).sigma_y(k)) + ...
                        lognormpdf(poseData(i, b, 2), dot(pval, P.clg(b).theta(k, 5:8)), P.clg(b).sigma_x(k)) + ...
                        lognormpdf(poseData(i, b, 3), dot(pval, P.clg(b).theta(k, 9:end)), P.clg(b).sigma_angle(k));                
                end
                logClassJointProb(i, k) = logClassJointProb(i, k) + logPartProb;
            end
        end
    end
    ClassProb = exp(logClassJointProb - repmat(logsumexp(logClassJointProb), 1, K));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute log likelihood of dataset for this iteration
    % Hint: You should use the logsumexp() function here
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loglikelihood(iter) = sum(logsumexp(logClassJointProb));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Print out loglikelihood
    disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end

    % Check for overfitting: when loglikelihood decreases
    if iter > 1
        if loglikelihood(iter) < loglikelihood(iter-1)
            break;
        end
    end

end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
