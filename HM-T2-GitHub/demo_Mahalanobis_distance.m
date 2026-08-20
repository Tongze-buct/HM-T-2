%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HM-T^2 demo on the Tennessee Eastman process
%
% Authors: Tongze Hou, Xin Ma, Haotian Zhang, and Youqing Wang
%
% Companion implementation for:
% "Prior Probability Can Enhance Hotelling's T-square Index to Improve
% Process Monitoring Performance"
%
% The script uses Mahalanobis distance (MD) as the preceding monitoring
% method and processes the test data sample by sample to simulate online
% monitoring.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(10)
clc
clear

%% Data
% d00_te.dat: normal training data
% d15_te.dat: TE fault IDV(15)
Train = load('te/d00_te.dat');
Test = load('te/d09_te.dat');
kesai = 0.01;

% 22 measured variables + 11 manipulated variables.
Train = Train(:,[1:22,42:52]);
Test = Test(:,[1:22,42:52]);
v = size(Train,2);

%% Stationarity test
% The stationarity flag controls whether the dynamic transition-probability
% update is activated. For KPI-related monitoring methods, set flag = 1
% manually according to the HM-T^2 implementation used in the paper.
h_adf = zeros(1,size(Train,2));
p_adf = zeros(1,size(Train,2));
for i = 1:size(Train,2)
    [h_adf(i),p_adf(i)] = my_adftest(Train(:,i));
end

flag = 0;
if all(h_adf == 0)
    flag = 1;
end

%% MD model training
% Estimate the MD model once from fault-free training data. Each test sample
% is then mapped to a single T^2 value using the fixed model parameters.
[T2_Train,MD_model] = MD(Train);

%% HM-T^2 parameters
% Pretrained parameters are included with the repository. They can be
% replaced by parameters obtained from the parameter-selection procedure
% described in the paper.
% To retrain the parameters, simply delete the corresponding .mat files.
if exist('slice_op.mat','file') == 2 && exist('u_op.mat','file') == 2
    load('slice_op.mat')
    load('u_op.mat')
else
    [slice_op,u_op] = para_setting(T2_Train,v);
end

if exist('A.mat','file') == 2 && exist('B.mat','file') == 2
    load('A.mat')
    load('B.mat')
else
    [A,B] = HMT2_train(slice_op,u_op,T2_Train,v);
end

%% Real-time monitoring
% At sampling instant k:
%   1. MD receives the new process sample x_k and returns T^2_k.
%   2. T^2_k is appended to the available T^2 history.
%   3. HMT2_Test uses the history to infer the current HMM state and HM-T^2.
fp = 161;
N_test = size(Test,1);
w = N_test;

T2_Test = zeros(1,N_test);
HMT2_RT = zeros(1,N_test);
states_RT = zeros(1,N_test);

for k = 1:N_test
    T2_Test(k) = MD(Test(k,:),MD_model);

    T2_history = T2_Test(1:k);
    [states_RT(k),~,~,~,HMT2_RT(k)] = ...
        HMT2_Test(T2_history,[],slice_op,v,A,B,kesai,flag,w);
end

%% Performance evaluation
% FAR and FDR are calculated after the complete test stream is processed.
FAR = sum(states_RT(1:fp-1) > slice_op)/(fp-1)*100;
FDR = sum(states_RT(fp:end) > slice_op)/numel(states_RT(fp:end))*100;
% FAR = sum(HMT2_RT(1:fp-1) > 0)/(fp-1)*100;
% FDR = sum(HMT2_RT(fp:end) > 0)/numel(HMT2_RT(fp:end))*100;
Jth = ones(1,N_test)*chi2inv(0.99,v); 

fprintf('Real-time simulation finished.\n');
fprintf('FAR = %.2f%%\n',FAR);
fprintf('FDR = %.2f%%\n',FDR);
