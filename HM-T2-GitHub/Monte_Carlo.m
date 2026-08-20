function expected_cdf = Monte_Carlo(N,n,a)
%MONTE_CARLO Estimate the lower-tail probability of the fault distribution.
%
% The noncentrality parameter is sampled over the range used by the HM-T^2
% fault-observation model. The result is used to construct the fault-state
% emission probabilities in HMT2_train.

    R = 2*sqrt(chi2inv(0.99,n));
    lambda_samples = R^2*rand(N,1);
    cdf_values = ncx2cdf(a,n,lambda_samples);
    expected_cdf = mean(cdf_values);
end
