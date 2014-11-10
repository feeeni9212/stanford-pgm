% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

% featureSet is a struct with two fields:
%    .numParams - the number of parameters in the CRF (this is not numImageFeatures
%                 nor numFeatures, because of parameter sharing)
%    .features  - an array comprising the features in the CRF.
%
% Each feature is a binary indicator variable, represented by a struct 
% with three fields:
%    .var          - a vector containing the variables in the scope of this feature
%    .assignment   - the assignment that this indicator variable corresponds to
%    .paramIdx     - the index in theta that this feature corresponds to
%
% For example, if we have:
%   
%   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
%
% then feature is an indicator function over X_2 and X_3, which takes on a value of 1
% if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
% Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
%
% If you're interested in the implementation details of CRFs, 
% feel free to read through GenerateAllFeatures.m and the functions it calls!
% For the purposes of this assignment, though, you don't
% have to understand how this code works. (It's complicated.)

featureSet = GenerateAllFeatures(X, modelParams);

% Use the featureSet to calculate nll and grad.
% This is the main part of the assignment, and it is very tricky - be careful!
% You might want to code up your own numerical gradient checker to make sure
% your answers are correct.
%
% Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
%       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
%       if you want to use it to compute logZ.

% Your code here:
F = features2factors(featureSet, y, theta, modelParams);
P = CreateCliqueTree(F);
[P, logZ] = CliqueTreeCalibrate(P, false);

% normalize each clique in the calibrated clique tree
for i = 1:length(P.cliqueList)
    P.cliqueList(i).val = P.cliqueList(i).val / sum(P.cliqueList(i).val);
end

[weightedFeatureCounts, dataFeatureCounts, modelFeatureCounts] = ...
    computeFeatureCounts(featureSet, y, P, theta);

% compute nll
regCost = modelParams.lambda/2 * dot(theta, theta);
nll = logZ - sum(weightedFeatureCounts) + regCost;

% compute gradient of nll
regGrad = modelParams.lambda * theta;
grad = modelFeatureCounts - dataFeatureCounts + regGrad;

end


function Fnew = features2factors(featureSet, y, theta, modelParams)
%
F = initFactors(y, modelParams);
Fnew = [];

for i = 1:length(F)
    factor = F(i);
    for j = 1:length(featureSet.features)
        f = featureSet.features(j);
        if sameScope(f, factor)
            assign = f.assignment;
            factor = SetValueOfAssignment(factor, assign, ...
                GetValueOfAssignment(factor, assign)+theta(f.paramIdx));
        end
    end
    factor.val = exp(factor.val);
    Fnew = [Fnew factor];
end

end


function F = initFactors(y, modelParams)
%
N = length(y);  % number of character labels
F = repmat(struct('var', [], 'card', [], 'val', []) , 1, 2*N-1);
v = modelParams.numHiddenStates;

for i = 1:N
    F(i).var = i;
    F(i).card = v;
    F(i).val = zeros(1, v);
    
    if i == N, continue; end
    
    F(i+N).var = [i i+1];
    F(i+N).card = [v v];
    F(i+N).val = zeros(1, v*v);
end

end


function same = sameScope(f1, f2)
%   same = sameScope(f1, f2)
% 
% Checks if the two features (or factors), f1 and f2, have the same scope.
% This function works on combinations of features/factors.
same = length(f1.var) == length(f2.var) && all(f1.var == f2.var);
end


function [weightedFeatureCounts, dataFeatureCounts, modelFeatureCounts] = ...
    computeFeatureCounts(featureSet, y, P, theta)
% Cliques in P are assumed to be normalized.

M = extractMarginals(P);    % marginals of each variable

weightedFeatureCounts   = zeros(size(theta));
dataFeatureCounts       = zeros(size(theta));
modelFeatureCounts   = zeros(size(theta));

for i = 1:length(featureSet.features)
    f = featureSet.features(i);
    indx = f.paramIdx;
    
    % add contributions to the model feature counts
    if length(f.var) == 1   % singleton feature
        factor = M(f.var);
    else                    % pair feature
        for k = 1:length(P.cliqueList)
            if all(ismember(f.var, P.cliqueList(k).var))
                % locate the clique containing the feature vars
                factor = P.cliqueList(k);
                factor = FactorMarginalization(factor, setdiff(factor.var, f.var));
            end
        end
    end
    modelFeatureCounts(indx) = modelFeatureCounts(indx) + GetValueOfAssignment(factor, f.assignment);
    
    % add contributions to the empirical weighted and data feature counts
    % if data instance matches feature assignment
    addCount = true;
    for j = 1:length(f.var)
        if y(f.var(j)) ~= f.assignment(j)
            addCount = false;
            break;
        end
    end
    if addCount
        weightedFeatureCounts(indx) = weightedFeatureCounts(indx) + theta(indx);
        dataFeatureCounts(indx) = dataFeatureCounts(indx) + 1;
    end
end

end


function M = extractMarginals(P)
% extract marginals from calibrated clique tree P
N = length(unique([P.cliqueList(:).var]));    % total number of variables
M = repmat(struct('var', [], 'card', [], 'val', []), N, 1);

for i = 1:N
    for j = 1:length(P.cliqueList)
        if ismember(i, P.cliqueList(j).var)
            M(i) = FactorMarginalization(P.cliqueList(j), setdiff(P.cliqueList(j).var, i));
            M(i).val = M(i).val / sum(M(i).val);
            break;
        end
    end
end
end
