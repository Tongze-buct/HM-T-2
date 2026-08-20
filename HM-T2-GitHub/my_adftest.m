function [h,pValue] = my_adftest(y)
%MY_ADFTEST Lightweight ADF-type stationarity test without Econometrics Toolbox.
%
% The regression is
%       Delta y_t = rho*y_{t-1} + e_t
% with no constant, trend, or additional lag terms.
%
% H0: the series contains a unit root.
% H1: the series is stationary.
%
% h = 1 indicates rejection of H0 at the approximate 5% level.
% h = 0 indicates that H0 is not rejected.
%
% The critical values and p-value interpolation below are asymptotic
% approximations. This helper is used only to reproduce the stationarity
% pre-check in the demo.

    y = y(:);
    y = y(~isnan(y));

    dy = diff(y);
    ylag = y(1:end-1);

    rho = ylag\dy;
    residual = dy-ylag*rho;

    n = length(dy);
    sigma2 = sum(residual.^2)/(n-1);
    se = sqrt(sigma2/sum(ylag.^2));
    stat = rho/se;

    cv_1 = -2.566;
    cv_5 = -1.941;
    cv_10 = -1.617;

    h = stat < cv_5;

    if stat <= cv_1
        pValue = 0.01;
    elseif stat <= cv_5
        pValue = 0.01 + (stat-cv_1)/(cv_5-cv_1)*(0.05-0.01);
    elseif stat <= cv_10
        pValue = 0.05 + (stat-cv_5)/(cv_10-cv_5)*(0.10-0.05);
    else
        pValue = min(1,0.10 + (stat-cv_10)*0.20);
    end
end
