%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = C.edges;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% retrieve all variable cards from the factor list
factors = C.factorList;
varCard = zeros(1, length(unique([factors.var])));
for i = 1:length(varCard)
    for j = 1:length(factors)
        indx = find(factors(j).var == i); % the index of i in factor j, empty if i is not in factor j
        if ~isempty(indx)
            varCard(i) = factors(j).card(indx);
            break;
        end
    end
end

% assign each factor to the first clique that can accommodate it, and
% compute the product of these factors.
F = 1:length(factors);   % indices of unassigned factors
for i = 1:N
    P.cliqueList(i).var = C.nodes{i};
    P.cliqueList(i).card = varCard(P.cliqueList(i).var);
    P.cliqueList(i).val = ones(1, prod(P.cliqueList(i).card));
    for f = F
        if all(ismember(factors(f).var, P.cliqueList(i).var))
            P.cliqueList(i) = FactorProduct(P.cliqueList(i), factors(f));
            F = setdiff(F, f);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

