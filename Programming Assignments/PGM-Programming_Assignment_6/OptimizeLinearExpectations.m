% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
% Inputs: An influence diagram I with a single decision node and one or more utility nodes.
%         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
%              the child variable = D.var(1)
%         I.DecisionFactors = factor for the decision node.
%         I.UtilityFactors = list of factors representing conditional utilities.
% Return value: the maximum expected utility of I and an optimal decision rule 
% (represented again as a factor) that yields that expected utility.
% You may assume that there is a unique optimal decision.
%
% This is similar to OptimizeMEU except that we will have to account for
% multiple utility factors.  We will do this by calculating the expected
% utility factors and combining them, then optimizing with respect to that
% combined expected utility factor.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
%
% A decision rule for D assigns, for each joint assignment to D's parents, 
% probability 1 to the best option from the EUF for that joint assignment 
% to D's parents, and 0 otherwise.  Note that when D has no parents, it is
% a degenerate case we can handle separately for convenience.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% separately compute each expected utility factor and add them up
IPrime = I;
EUF = struct('var', [], 'card', [], 'val', []);
for i = 1:length(I.UtilityFactors)
    IPrime.UtilityFactors = [I.UtilityFactors(i)];
    EUF = FactorSum(EUF, CalculateExpectedUtilityFactor(IPrime));
end

% compute the optimal decision rule
D = I.DecisionFactors(1);
D.val = zeros(1, prod(D.card));
[Pa, mapPa] = setdiff(EUF.var, D.var(1));   % parents of decision node D

if isempty(Pa)      % if D has no parents
    [~, maxIndx] = max(EUF.val);
    D.val(maxIndx) = 1;
else                % otherwise, find an optimal action for each parent assignment
    assignments = IndexToAssignment(1:length(EUF.val), EUF.card);
    PaIndx = AssignmentToIndex(assignments(:, mapPa), EUF.card(mapPa));
    for i = unique(PaIndx)'
        [~, maxIndx] = max(EUF.val(PaIndx == i));
        PaAssign = IndexToAssignment(i, EUF.card(mapPa));
        D = SetValueOfAssignment(D, [maxIndx PaAssign], 1);
    end
end

MEUF = FactorProduct(D, EUF);
OptimalDecisionRule = D;
MEU = sum(MEUF.val);

end
