function transitionMatrix = transition(states)
%TRANSITION Estimate a Markov transition matrix from a state sequence.
%
% transitionMatrix(i,j) is the empirical probability of a transition from
% state i to state j. Rows without observed outgoing transitions are set to 0.

    numStates = max(states);
    transitionCounts = zeros(numStates,numStates);

    for i = 1:length(states)-1
        from = states(i);
        to = states(i+1);
        transitionCounts(from,to) = transitionCounts(from,to) + 1;
    end

    transitionMatrix = transitionCounts./sum(transitionCounts,2);
    transitionMatrix(isnan(transitionMatrix)) = 0;
end
