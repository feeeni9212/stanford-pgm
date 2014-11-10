%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
N = length(unique([F.var]));    % total number of variables
M = repmat(struct('var', [], 'card', [], 'val', []), N, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isMax
    FactorOp = @FactorMaxMarginalization;
else
    FactorOp = @FactorMarginalization;
end

T = CreateCliqueTree(F, E);
T = CliqueTreeCalibrate(T, isMax);

for i = 1:N
    for j = 1:length(T.cliqueList)
        if ismember(i, T.cliqueList(j).var)
            M(i) = FactorOp(T.cliqueList(j), setdiff(T.cliqueList(j).var, i));
            if ~isMax, M(i).val = M(i).val / sum(M(i).val); end
            break;
        end
    end
end
end
