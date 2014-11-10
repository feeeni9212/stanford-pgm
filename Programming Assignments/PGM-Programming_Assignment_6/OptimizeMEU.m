% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

% Inputs: An influence diagram I with a single decision node and a single utility node.
%         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
%              the child variable = D.var(1)
%         I.DecisionFactors = factor for the decision node.
%         I.UtilityFactors = list of factors representing conditional utilities.
% Return value: the maximum expected utility of I and an optimal decision rule 
% (represented again as a factor) that yields that expected utility.

% We assume I has a single decision node.
% You may assume that there is a unique optimal decision.
D = I.DecisionFactors(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE...
% 
% Some other information that might be useful for some implementations
% (note that there are multiple ways to implement this):
% 1.  It is probably easiest to think of two cases - D has parents and D 
%     has no parents.
% 2.  You may find the Matlab/Octave function setdiff useful.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D.val = zeros(1, prod(D.card));
EUF = CalculateExpectedUtilityFactor(I);    % expected utility factor
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
