% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = length(datasetTrain);   % number of types of actions
Param = [];                 % learned parameters for each HMM model

for k = 1:K
    data = datasetTrain(k);
    [P, ~, ~, ~] = EM_HMM(data.actionData, data.poseData, G,...
        data.InitialClassProb, data.InitialPairProb, maxIter);
    Param = [Param, P];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTest
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

A = length(datasetTest.actionData); % number of test actions
N = size(datasetTest.poseData, 1);  % number of poses in all actions
predicted_labels = zeros(A, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute logEmissionProb
logEmissionProbs = zeros(K, N, K);
for t = 1:K
    P = Param(t);
    for i = 1:N
        for k = 1:K
            for b = 1:size(G, 1)
                if G(b, 1) == 0
                    % only class variable as parent
                    logPartProb = lognormpdf(datasetTest.poseData(i, b, 1), P.clg(b).mu_y(k), P.clg(b).sigma_y(k)) + ...
                        lognormpdf(datasetTest.poseData(i, b, 2), P.clg(b).mu_x(k), P.clg(b).sigma_x(k)) + ...
                        lognormpdf(datasetTest.poseData(i, b, 3), P.clg(b).mu_angle(k), P.clg(b).sigma_angle(k));
                else
                    % body part b has another parent, besides class variable
                    p = G(b, 2);   % parent body part
                    pval = [1; datasetTest.poseData(i, p, 1); datasetTest.poseData(i, p, 2); datasetTest.poseData(i, p, 3)];
                    logPartProb = lognormpdf(datasetTest.poseData(i, b, 1), dot(pval, P.clg(b).theta(k, 1:4)), P.clg(b).sigma_y(k)) + ...
                        lognormpdf(datasetTest.poseData(i, b, 2), dot(pval, P.clg(b).theta(k, 5:8)), P.clg(b).sigma_x(k)) + ...
                        lognormpdf(datasetTest.poseData(i, b, 3), dot(pval, P.clg(b).theta(k, 9:end)), P.clg(b).sigma_angle(k));                
                end
                logEmissionProbs(t, i, k) = logEmissionProbs(t, i, k) + logPartProb;
            end
        end
    end
end


logLikelihoods = zeros(K, 1);
for a = 1:A    
    % compute the loglikelihood of the action for each type of HMM
    for t = 1:K
        P = Param(t);
        
        % extract the corresponding logEmissionProb
        logEmissionProb = squeeze(logEmissionProbs(t, :, :));
        
        % run inference to compute likelihood of action using P
        initialStateF.var = 1;
        initialStateF.card = K;
        initialStateF.val = log(P.c);
        logTransProb = reshape(log(P.transMatrix), 1, []);
        
        poseCondFactors = repmat(struct('var', [], 'card', [K], 'val', []), 1,...
            length(datasetTest.actionData(a).marg_ind));
        transFactors = repmat(struct('var', [], 'card', [K K], 'val', []), 1,...
            length(datasetTest.actionData(a).pair_ind));
        
        for i = 1:length(datasetTest.actionData(a).marg_ind)
            poseIndx = datasetTest.actionData(a).marg_ind(i);
            poseCondFactors(i).var = i;
            poseCondFactors(i).val = logEmissionProb(poseIndx, :);
        end
        
        for j = 1:length(datasetTest.actionData(a).pair_ind)
            transFactors(j).var = [j j+1];
            transFactors(j).val = logTransProb;
        end
        
        % run inference to compute marginals and calibrated cliques
        Factors = [initialStateF, transFactors, poseCondFactors];
        [~, CT] = ComputeExactMarginalsHMM(Factors);
        
       logLikelihoods(t) = logsumexp(CT.cliqueList(1).val);
    end
    
    % predict labels
    [~, indx] = max(logLikelihoods);
    predicted_labels(a) = indx;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accuracy = sum(predicted_labels == datasetTest.labels) / A;
