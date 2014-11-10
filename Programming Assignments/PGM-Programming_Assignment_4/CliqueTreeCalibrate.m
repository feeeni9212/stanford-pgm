%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)

if isMax
    FactorOp1 = @FactorSum;
    FactorOp2 = @FactorMaxMarginalization;
else
    FactorOp1 = @FactorProduct;
    FactorOp2 = @FactorMarginalization;
end

% Number of cliques in the tree.
N = length(P.cliqueList);

% log-transform the cliques if doing MAP inference
if isMax
    for i = 1:N, P.cliqueList(i).val = log(P.cliqueList(i).val); end;
end

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[i, j] = GetNextCliques(P, MESSAGES);
while all([i j])
    % compute message to be sent from i to j
    message = P.cliqueList(i);  % initialize to the initial belif of i
    for k = setdiff(find(P.edges(i, :)), j)
        message = FactorOp1(message, MESSAGES(k, i));
    end
    % (max)-marginalize out irrelevant variables
    varToSumOut = setdiff(message.var, P.cliqueList(j).var);
    message = FactorOp2(message, varToSumOut);
    % normalize message if not doing MAP inference
    if ~isMax, message.val = message.val / sum(message.val); end;
    
    MESSAGES(i, j) = message;
    [i, j] = GetNextCliques(P, MESSAGES);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(P.cliqueList)
    for j = find(P.edges(i, :))
        P.cliqueList(i) = FactorOp1(P.cliqueList(i), MESSAGES(j, i));
    end
end

end


function C = FactorSum(A, B)
% C = FactorProduct(A,B) computes the sum of two factors, A and B.
% The code here is highly adapted from the FactorProduct function by Prof.
% Daphne Koller.

% Check for empty factors
if (isempty(A.var)), C = B; return; end;
if (isempty(B.var)), C = A; return; end;

% Check that variables in both A and B have the same cardinality
[dummy iA iB] = intersect(A.var, B.var);
if ~isempty(dummy)
	% A and B have at least 1 variable in common
	assert(all(A.card(iA) == B.card(iB)), 'Dimensionality mismatch in factors');
end

% Set the variables of C
C.var = union(A.var, B.var);

[~, mapA] = ismember(A.var, C.var);
[~, mapB] = ismember(B.var, C.var);

% Set the cardinality of variables in C
C.card = zeros(1, length(C.var));
C.card(mapA) = A.card;
C.card(mapB) = B.card;

% Initialize the factor values of C:
%   prod(C.card) is the number of entries in C
C.val = zeros(1,prod(C.card));

% Compute some helper indices
% These will be very useful for calculating C.val
% so make sure you understand what these lines are doing.
assignments = IndexToAssignment(1:prod(C.card), C.card);
indxA = AssignmentToIndex(assignments(:, mapA), A.card);
indxB = AssignmentToIndex(assignments(:, mapB), B.card);

C.val = A.val(indxA) + B.val(indxB);
end
