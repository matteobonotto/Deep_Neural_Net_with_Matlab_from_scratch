function X_test = fun_featureNormalize_test(net,x_test)

X = x_test.';
X = (X - repmat(net.X.nnMu,size(X,1),1))...
        ./repmat(net.X.nnSigma,size(X,1),1);
X_test = X.';
