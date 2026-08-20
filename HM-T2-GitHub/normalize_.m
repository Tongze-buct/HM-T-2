function [X_norm,Y_norm,mean0,sigma0] = normalize_(X,Y)
%NORMALIZE_ Standardize data using the mean and standard deviation of X.
%
%   X_norm = normalize_(X)
%   [X_norm,Y_norm,mean0,sigma0] = normalize_(X,Y)
%
% When Y is supplied, the normalization parameters estimated from X are
% applied to both X and Y.

    x_mean = mean(X);
    x_std = std(X);
    [x_row,~] = size(X);
    X_norm = (X-repmat(x_mean,x_row,1))./repmat(x_std,x_row,1);

    if nargin == 2
        y_row = size(Y,1);
        Y_norm = (Y-repmat(x_mean,y_row,1))./repmat(x_std,y_row,1);
    else
        Y_norm = [];
    end

    mean0 = x_mean;
    sigma0 = x_std;
end
