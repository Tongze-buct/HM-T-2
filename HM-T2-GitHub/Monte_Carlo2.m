function expected_cdf = Monte_Carlo2(N,n,a)
%MONTE_CARLO2 Estimate the upper-tail probability of the fault distribution.
%
% This function evaluates the probability above threshold a for the same
% noncentral chi-square mixture used in HMT2_train.

    R = 2*sqrt(chi2inv(0.99,n));
    lambda_samples = R^2*rand(N,1);
    cdf_values = 1-ncx2cdf(a,n,lambda_samples);
    expected_cdf = mean(cdf_values);
end
