function [X,T] = fun_featureNormalize_training(x_dataset,t_dataset)

%%% Normalize training dataset

temp_X = x_dataset.';
X_norm = zeros(size(temp_X));
X_norm(:,1) = ones(size(temp_X,1),1);
[X_norm(:,1:end), nnMu, nnSigma] = featureNormalize(temp_X(:,1:end));
x_dataset = X_norm.';

X.data = x_dataset;
X.nnMu = nnMu;
X.nnSigma = nnSigma;
% X.nnMu = 0;
% X.nnSigma = 1;


%%% Normalize target dataset

temp_X = t_dataset.';
X_norm = zeros(size(temp_X));
X_norm(:,1) = ones(size(temp_X,1),1);
[X_norm(:,1:end), nnMu, nnSigma] = featureNormalize(temp_X(:,1:end));
t_dataset = X_norm.';

T.data = t_dataset;
T.nnMu = nnMu;
T.nnSigma = nnSigma;
% % T.nnMu = 0;
% % T.nnSigma = 1;



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.
%
%   usage: [X_norm, mu, sigma] = FEATURENORMALIZE(X)

% column-wise mean
mu = mean(X);

% column-wise standard deviation
sigma = std(X);

% normalized matrix
X_norm = (X - repmat(mu,size(X, 1), 1))./repmat(sigma,size(X, 1), 1);
X_norm(isnan(X_norm)) = 0;

end
