function varargout = MD(X,model_or_test)
%MD Mahalanobis-distance-based T^2 statistic.
%
% Model training:
%   [T2_Train,model] = MD(Train);
%
% Real-time evaluation:
%   T2_k = MD(x_k,model);
%
% Batch evaluation:
%   [T2_Train,T2_Test] = MD(Train,Test);
%
% The training data are standardized first. The covariance matrix of the
% standardized variables is decomposed by SVD, and the resulting quadratic
% form is used as the T^2 statistic. The model parameters are stored once so
% that incoming samples can be evaluated without retraining.

    if nargin == 1
        [T2,model] = fit_md_model(X);
        varargout{1} = T2;
        if nargout >= 2
            varargout{2} = model;
        end
        return
    end

    if isstruct(model_or_test)
        model = model_or_test;
        T2 = evaluate_md(X,model);
        varargout{1} = T2;
        if nargout >= 2
            varargout{2} = model;
        end
        return
    end

    % Batch interface retained for comparison with the original implementation.
    Train = X;
    Test = model_or_test;
    [T2_Train,model] = fit_md_model(Train);
    T2_Test = evaluate_md(Test,model);

    varargout{1} = T2_Train;
    if nargout >= 2
        varargout{2} = T2_Test;
    end
    if nargout >= 3
        varargout{3} = model;
    end
end


function [T2,model] = fit_md_model(Train)
%FIT_MD_MODEL Estimate normalization and covariance parameters from Train.

    x_mean = mean(Train,1);
    x_std = std(Train,0,1);

    if any(x_std == 0)
        error('MD:ZeroVariance', ...
            'At least one training variable has zero standard deviation.');
    end

    x_norm = (Train-x_mean)./x_std;
    [U,Sigma] = svd(cov(x_norm));

    model.mean = x_mean;
    model.std = x_std;
    model.U = U;
    model.SigmaInv = inv(Sigma);

    score = x_norm*model.U;
    T2 = sum((score*model.SigmaInv).*score,2).';
end


function T2 = evaluate_md(X,model)
%EVALUATE_MD Evaluate one or more samples using a trained MD model.

    if size(X,2) ~= numel(model.mean)
        error('MD:DimensionMismatch', ...
            'The number of variables in X does not match the MD model.');
    end

    x_norm = (X-model.mean)./model.std;
    score = x_norm*model.U;
    T2 = sum((score*model.SigmaInv).*score,2).';
end
