function [currentState,logP,v] = hmmviterbi0(seq,tr,e)
%HMMVITERBI0 Viterbi decoding for the HM-T^2 monitoring model.
%
%   [currentState,logP,v] = hmmviterbi0(seq,tr,e)
%
% seq           Discrete observation sequence.
% tr            State-transition probability matrix.
% e             Emission probability matrix.
% currentState  Most probable state path.
% logP          Log score of the most probable final path.
% v             Final Viterbi log score of each state.
%
% The recursion is evaluated in the log domain. The model is initialized in
% state 1 before the first observation, consistent with the HM-T^2 training
% and testing code in this repository.

    if size(tr,1) ~= size(tr,2)
        error('hmmviterbi0:BadTransitions', ...
            'The transition matrix must be square.');
    end

    numStates = size(tr,1);
    if size(e,1) ~= numStates
        error('hmmviterbi0:InputSizeMismatch', ...
            'The emission matrix must have one row for each state.');
    end

    numEmissions = size(e,2);
    seq = seq(:).';
    L = numel(seq);

    if any(seq < 1) || any(seq ~= round(seq)) || any(seq > numEmissions)
        error('hmmviterbi0:BadSequence', ...
            'The observation sequence contains an invalid emission index.');
    end

    if L == 0
        currentState = [];
        logP = -Inf;
        v = -Inf(numStates,1);
        return
    end

    logTR = log(tr);
    logE = log(e);

    backPtr = zeros(numStates,L);
    vOld = -Inf(numStates,1);
    vOld(1) = 0;

    for t = 1:L
        v = -Inf(numStates,1);
        for state = 1:numStates
            [bestVal,bestState] = max(vOld + logTR(:,state));
            v(state) = logE(state,seq(t)) + bestVal;
            backPtr(state,t) = bestState;
        end
        vOld = v;
    end

    [logP,finalState] = max(v);

    currentState = zeros(1,L);
    currentState(L) = finalState;
    for t = L-1:-1:1
        currentState(t) = backPtr(currentState(t+1),t+1);
    end
end
