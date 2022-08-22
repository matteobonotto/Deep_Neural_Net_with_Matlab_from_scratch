function out = fun_denormalize_output(out,net)

X = out.';
X = (X.*repmat(net.T.nnSigma,size(X,1),1) + repmat(net.T.nnMu,size(X,1),1));
out = X.';