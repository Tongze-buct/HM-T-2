function [A,B] = HMT2_train(slice_op,u_op,T2_Train,v)
%HMT2_TRAIN Estimate the HMM parameters used by HM-T^2.
%
%   [A,B] = HMT2_train(slice_op,u_op,T2_Train,v)
%
% slice_op  Number of normal substates.
% u_op      Emission-probability smoothing parameter.
% T2_Train  T^2 statistics from fault-free training data.
% v         Degrees of freedom of the chi-square distribution.
%
% The HMM contains 'slice_op' normal substates and one fault state. Normal
% substates are defined by chi-square quantiles. Normal-to-normal transition
% probabilities are estimated from the discretized training sequence, and
% the normal-to-fault probability increases with the normal-state index.
%
% The fault-state emission probabilities are approximated using a mixture of
% noncentral chi-square distributions. Monte Carlo integration is used for
% the corresponding interval probabilities.

    u = u_op;
    slice = slice_op;
    T0 = T2_Train;
    row = v;
    S = 1:slice;

    %% Discretize the normal training sequence
    S_limit = zeros(1,length(S)-1);
    for i = 1:length(S)-1
        S_limit(i) = chi2inv(i/length(S),row);
    end
    S_limit = [0 S_limit inf];
    S_ = discretize(T0,S_limit);

    %% Transition probability matrix A
    % The final state is the fault state. No transition from the fault state
    % back to a normal state is allowed in the monitoring model.
    p_nf = 0.01*linspace(1,length(S),length(S))';
    A = [transition(S_) p_nf; zeros(1,length(S)) 1];

    A(1:end-1,1:end-1) = ...
        diag(ones(1,length(S))-p_nf')*A(1:end-1,1:end-1) ...
        ./sum(A(1:end-1,1:end-1),2);

    %% Fault-state emission probabilities
    expected_cdf = [];
    for i = 1:length(S)-1
        expected_cdf = [expected_cdf ...
            Monte_Carlo(10000,row,chi2inv(i/length(S),row))]; %#ok<AGROW>
    end
    expected_cdf = [expected_cdf ...
        Monte_Carlo(10000,row,chi2inv(0.99,row)) ...
        Monte_Carlo2(10000,row,chi2inv(0.99,row))];
    expected_cdf = [0 expected_cdf];

    interval_prob = zeros(1,length(expected_cdf)-1);
    for i = 1:length(expected_cdf)-1
        if i < length(expected_cdf)-1
            interval_prob(i) = expected_cdf(i+1)-expected_cdf(i);
        else
            interval_prob(i) = expected_cdf(i+1);
        end
    end
    interval_prob = interval_prob./sum(interval_prob,2);

    %% Emission probability matrix B
    % u controls the smoothing between an ideal substate-emission matrix and
    % a distance-based emission matrix for the normal states.
    P = zeros(length(S),length(S)+1);
    for i = 1:length(S)
        for j = 1:length(S)+1
            P(i,j) = exp(-(j-i)^2);
        end
    end
    P = P./sum(P,2);

    Pnew = ([eye(length(S)) zeros(length(S),1)] + u*P)./(1+u);
    B = [Pnew; interval_prob];

    save('A.mat','A');
    save('B.mat','B');
end
