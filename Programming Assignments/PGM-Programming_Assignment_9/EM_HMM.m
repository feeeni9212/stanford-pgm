% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2);        % number of actions
V = size(InitialPairProb, 1);
Gk = G;
B = size(G, 1);                 % number of body parts

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

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
    % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
    % Fill in P.clg for each body part and each class
    % Make sure to choose the right parameterization based on G(i,1)
    % Hint: This part should be similar to your work from PA8 and EM_cluster.m

    P.c = zeros(1,K);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for a = 1:L
        P.c = P.c + ClassProb(actionData(a).marg_ind(1), :);
    end
    P.c = P.c / L;

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

    % M-STEP to estimate parameters for transition matrix
    % Fill in P.transMatrix, the transition matrix for states
    % P.transMatrix(i,j) is the probability of transitioning from state i to state j
    P.transMatrix = zeros(K,K);

    % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
    P.transMatrix = P.transMatrix + size(PairProb,1) * .05;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     for l = 1:L
%         for i = 1:length(actionData(l).pair_ind)
%             pairIndx = actionData(l).pair_ind(i);
%             P.transMatrix = P.transMatrix + reshape(PairProb(pairIndx, :), [K K]);
%         end
%     end
%     normalization = sum(P.transMatrix, 2);
%     P.transMatrix = P.transMatrix ./ repmat(normalization, 1, 3);
    
    for p = 1:size(PairProb, 1)
        P.transMatrix = P.transMatrix + reshape(PairProb(p, :), [K K]);
    end
    normalization = sum(P.transMatrix, 2);
    P.transMatrix = P.transMatrix ./ repmat(normalization, 1, 3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
    % of the poses in all actions = log( P(Pose | State) )
    % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m

    logEmissionProb = zeros(N,K);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                logEmissionProb(i, k) = logEmissionProb(i, k) + logPartProb;
            end
        end
    end
    
    % code for debugging purpose
%     disp(logEmissionProb(1:5, :));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % E-STEP to compute expected sufficient statistics
    % ClassProb contains the conditional class probabilities for each pose in all actions
    % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
    % Also compute log likelihood of dataset for this iteration
    % You should do inference and compute everything in log space, only converting to probability space at the end
    % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues

    ClassProb = zeros(N,K);
    PairProb = zeros(V,K^2);
    loglikelihood(iter) = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    initialStateF.var = 1;
    initialStateF.card = K;
    initialStateF.val = log(P.c);
    logTransProb = reshape(log(P.transMatrix), 1, []);
    for a = 1:L
        poseCondFactors = repmat(struct('var', [], 'card', [K], 'val', []), 1,...
            length(actionData(a).marg_ind));
        transFactors = repmat(struct('var', [], 'card', [K K], 'val', []), 1,...
            length(actionData(a).pair_ind));
        
        for i = 1:length(actionData(a).marg_ind)
            poseIndx = actionData(a).marg_ind(i);
            poseCondFactors(i).var = i;
            poseCondFactors(i).val = logEmissionProb(poseIndx, :);
        end
        
        for j = 1:length(actionData(a).pair_ind)
            transFactors(j).var = [j j+1];
            transFactors(j).val = logTransProb;
        end
        
        % run inference to compute marginals and calibrated cliques
        Factors = [initialStateF, transFactors, poseCondFactors];
        [M, CT] = ComputeExactMarginalsHMM(Factors);
        
        % extract class conditional probs for each pose from marginals M
        for m = 1:length(M)
            poseIndx = actionData(a).marg_ind(M(m).var);
            ClassProb(poseIndx, :) = M(m).val;
        end
        
        % extract pairwise transition probs for each pair from cliques CT
        for c = 1:length(CT.cliqueList)
            clique = CT.cliqueList(c);
            pairIndx = actionData(a).pair_ind(clique.var(1));
            PairProb(pairIndx, :) = clique.val;
        end
        
        % add the loglikelihood of this action
        loglikelihood(iter) = loglikelihood(iter) + logsumexp(clique.val);
    end
    
    % normalize and exponentiate the probabilities
    ClassProb = exp(ClassProb - repmat(logsumexp(ClassProb), 1, K));
    PairProb = exp(PairProb - repmat(logsumexp(PairProb), 1, K^2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Print out loglikelihood
    disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end

    % Check for overfitting by decreasing loglikelihood
    if iter > 1
        if loglikelihood(iter) < loglikelihood(iter-1)
            break;
        end
    end

end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
