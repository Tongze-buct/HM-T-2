function [slice_op,u_op] = para_setting(T2,v)
%PARA_SETTING Select the HM-T^2 parameters using normal training data.
%
%   [slice_op,u_op] = para_setting(T2,v)
%
% T2        T^2 statistics from fault-free training data.
% v         Degrees of freedom of the chi-square distribution.
%
% slice_op  Selected number of normal substates.
% u_op      Selected emission-probability smoothing parameter.
%
% Candidate values of the number of normal substates and the smoothing
% parameter are evaluated using an artificially perturbed validation
% sequence. The parameter pair giving the minimum alarm rate is selected.
% The selected parameters are saved as slice_op.mat and u_op.mat.

    T0=T2;
    slice=linspace(1,10,10);
    u=linspace(0.1,1,10);
    row=v;

    %% Construct the perturbed validation sequence
    % Every 20th sample is shifted beyond the 99% chi-square limit.
    for k=1:length(T0)
        if rem(k,20)==0
           Tf(k)=0;
           while Tf(k)<chi2inv(0.99, row)
                Tf(k)=T0(k)+50*rand;
           end
        else
           Tf(k)=T0(k);
        end
    end

    %% Grid search over slice and u
    for i=1:length(slice)
        for j=1:length(u)
            S=linspace(1,slice(i),slice(i));

            % Divide the normal region into substates.
            S_limit=[];
            for l=1:length(S)-1
                S_limit(l)=chi2inv(l/length(S), row);
            end
            S_limit=[0 S_limit inf];

            for m=1:size(T0,2)
                segment = discretize(T0(m), S_limit);
                S_(m)=segment;
            end

            % Construct the transition probability matrix.
            A=[];
            A=[transition(S_(1:end)) 0.01*linspace(1,length(S),length(S))';[0/length(S)*ones(1,length(S)) 1]];
            A(1:end-1,1:end-1)=diag(ones(length(S),1)-0.01*linspace(1,length(S),length(S))')*A(1:end-1,1:end-1)./sum(A(1:end-1,1:end-1),2);

            % Estimate the fault-state emission probabilities.
            expected_cdf=[];
            for m=1:length(S)-1
                expected_cdf=[expected_cdf Monte_Carlo(10000,row,chi2inv(m/length(S), row))];
            end
            expected_cdf=[expected_cdf Monte_Carlo(10000,row,chi2inv(0.99, row)) Monte_Carlo2(10000,row,chi2inv(0.99, row))];
            expected_cdf=[0 expected_cdf];

            for k=1:length(expected_cdf)-1
                if k<length(expected_cdf)-1
                    aaa(k)=expected_cdf(k+1)-expected_cdf(k);
                else
                    aaa(k)=expected_cdf(k+1);
                end
            end
            expected_cdf=aaa;
            expected_cdf=expected_cdf./sum(expected_cdf,2);

            B=[eye(length(S)) zeros(length(S),1);
               expected_cdf ];

            % Smooth the normal-state emission probabilities.
            for m=1:length(S)
                for n=1:length(S)+1
                    P(m,n)=exp(-abs(n-m));
                end
            end
            P=P./sum(P,2);
            Pnew=([eye(length(S)) zeros(length(S),1)]+u(j)*P)./(1+u(j));
            B=[];
            B=[Pnew;expected_cdf];

            % Discretize the validation sequence.
            S_limit=[];
            for m=1:length(S)-1
                S_limit(m)=chi2inv(m/length(S), row);
            end
            S_limit=[0 S_limit chi2inv(0.99, row) inf];

            for m=1:size(Tf,2)
                segment = discretize(Tf(m), S_limit);
                S_f(m)=segment;
            end

            % Evaluate the alarm rate for the current parameter pair.
            states_f=[];
            for m=1:size(Tf,2)
                state = hmmviterbi(S_f(1:m), A, B);
                states_f=[states_f state(end)];
            end
            FAR(i,j)=length(find(states_f>slice(i)))/length(states_f);
        end
    end

    %% Select and save the optimal parameters
    [a,b]=find(FAR==min(min(FAR)));
    slice_op=max(a);
    index=find(a==slice_op);
    u_op=min(b(index))/10;

    save('slice_op.mat', 'slice_op');
    save('u_op.mat', 'u_op');
end
