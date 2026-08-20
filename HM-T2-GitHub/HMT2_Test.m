function [states_f,log_scores,FAR,FDR,HMT2] = HMT2_Test(T2,fault_point,slice,v,A,B,kesai,flag,w)
%HMT2_TEST HMM-enhanced T^2 (HM-T^2) monitoring.
%
% Batch mode:
%   [states,log_scores,FAR,FDR,HMT2] = ...
%       HMT2_Test(T2,fault_point,slice,v,A,B,kesai,flag);
%
% Real-time mode:
%   [state_k,~,~,~,HMT2_k] = ...
%       HMT2_Test(T2_history,[],slice,v,A,B,kesai,flag,w);
%
% T2          T^2 statistics available up to the current sample.
% fault_point First faulty sample in batch evaluation. Set [] for online use.
% slice       Number of normal substates in the HMM.
% v           Degrees of freedom of the chi-square distribution.
% A, B        HMM transition and emission probability matrices.
% kesai       Nominal false-alarm probability used by the prior update.
% flag        Set to 1 to enable transition-probability updating.
% w           Fixed reference length for the dynamic prior update.
%
% The normal region is divided into 'slice' substates according to chi-square
% quantiles. Samples above the 99% chi-square limit are assigned to the fault
% observation interval. The final HMM state is the fault state.
%
% HM-T^2 is evaluated from the current Viterbi log scores as the ratio between
% the fault-state score and the largest normal-state score. Therefore,
% HM-T^2 > 0 indicates that the fault state is more probable than any normal
% substate under the current state-transition history.

    Tf = T2(:).';
    row = v;

    if nargin < 9 || isempty(w)
        w = numel(Tf);
    end

    if isempty(Tf)
        states_f = [];
        log_scores = [];
        FAR = [];
        FDR = [];
        HMT2 = [];
        return
    end

    %% Discretize T^2 observations
    % Normal substates are separated by equal-probability chi-square intervals.
    % The last observation interval corresponds to T^2 > chi2inv(0.99,v).
    S_limit = zeros(1,slice-1);
    for i = 1:slice-1
        S_limit(i) = chi2inv(i/slice,row);
    end
    S_limit = [0 S_limit chi2inv(0.99,row) inf];
    S_f = discretize(Tf,S_limit);

    online_mode = isempty(fault_point);

    %% Real-time mode
    % Reconstruct the transition update from the observations available at k,
    % run Viterbi inference, and return only the current monitoring result.
    if online_mode
        A_current = update_transition_history(S_f,A,kesai,flag,w,slice);
        [state_path,~,log_score] = hmmviterbi0(S_f,A_current,B);

        states_f = state_path(end);
        log_scores = log_score(:).';
        HMT2 = log_score(end) - max(log_score(1:slice));
        FAR = [];
        FDR = [];
        return
    end

    %% Batch mode
    states_f = zeros(1,numel(Tf));
    log_scores = zeros(numel(Tf),size(A,1));
    HMT2 = zeros(1,numel(Tf));

    A0 = A;
    alpha_prev = 1;
    beta = 1;
    m1 = 0;
    m2 = 0;

    for i = 1:numel(Tf)
        if flag == 1
            % Dynamic prior update. m1 counts observations in the fault
            % interval and m2 counts all observations seen so far.
            m2 = m2 + 1;
            if S_f(i) == slice + 1
                m1 = m1 + 1;
            end

            alpha_now = max((kesai*w + m1)/(w + m2)/kesai,1);

            if A(end-1,end) < 0.95 && ...
                    (kesai*w + m1)/(w + m2)/kesai > 1 && ...
                    alpha_now > alpha_prev
                A(1:end-1,end) = alpha_now*A(1:end-1,end);
                A = A./sum(A,2);
                beta = alpha_now;
            end

            if alpha_now < alpha_prev
                A(1:end-1,end) = (1/beta)^(kesai/(1-kesai))*A(1:end-1,end);
                A = A./sum(A,2);
            end

            if A(end-1,end) < 0.01*slice
                A(1:end-1,end) = A0(1:end-1,end);
            end

            alpha_prev = alpha_now;
        end

        [state_path,~,log_score] = hmmviterbi0(S_f(1:i),A,B);
        states_f(i) = state_path(end);
        log_scores(i,:) = log_score(:).';
        HMT2(i) = exp(log_score(end) - max(log_score(1:slice)));
    end

    %% FAR and FDR
    if fault_point < 1 || fault_point > numel(states_f)
        FAR = [];
        FDR = [];
    else
        if fault_point > 1
            FAR = sum(states_f(1:fault_point-1) > slice)/(fault_point-1)*100;
        else
            FAR = [];
        end
        FDR = sum(states_f(fault_point:end) > slice)/numel(states_f(fault_point:end))*100;
    end
end


function A = update_transition_history(S_f,A,kesai,flag,w,slice)
%UPDATE_TRANSITION_HISTORY Replay the dynamic prior update up to time k.

    if flag ~= 1
        return
    end

    A0 = A;
    alpha_prev = 1;
    beta = 1;
    m1 = 0;
    m2 = 0;

    for i = 1:numel(S_f)
        m2 = m2 + 1;
        if S_f(i) == slice + 1
            m1 = m1 + 1;
        end

        alpha_now = max((kesai*w + m1)/(w + m2)/kesai,1);

        if A(end-1,end) < 0.95 && ...
                (kesai*w + m1)/(w + m2)/kesai > 1 && ...
                alpha_now > alpha_prev
            A(1:end-1,end) = alpha_now*A(1:end-1,end);
            A = A./sum(A,2);
            beta = alpha_now;
        end

        if alpha_now < alpha_prev
            A(1:end-1,end) = (1/beta)^(kesai/(1-kesai))*A(1:end-1,end);
            A = A./sum(A,2);
        end

        if A(end-1,end) < 0.01*slice
            A(1:end-1,end) = A0(1:end-1,end);
        end

        alpha_prev = alpha_now;
    end
end
