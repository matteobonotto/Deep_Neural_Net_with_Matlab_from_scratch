function net = fun_initialize_MLP(net,options)



%% Normalize datasets
[net.X,net.T] = fun_featureNormalize_training(net.x_dataset_train,net.t_dataset_train,options);


%%
net.n_in = size(net.X.data,1);
net.n_out = size(net.T.data,1);
net.m = size(net.T.data, 2); % Numeber of examples


if ~isfield(options, 'nw_hidden_layers')
    net.nw_hidden_layers = round(sqrt(n_out*n_in));
else
    net.nw_hidden_layers = options.nw_hidden_layers;
end
if ~isfield(options, 'f_activation')
    net.f_activation = 'logistic';
else
    net.f_activation = options.f_activation;
end
if ~isfield(options, 'nw_hidden_layers')
    net.threshold = eps;
else
    net.threshold = options.threshold;
end
if ~isfield(options, 'maxIter')
    net.threshold = 1000;
else
    net.maxIter = options.maxIter;
end



%%
net.nw_layers = [net.nw_hidden_layers net.n_out];
net.n_layers = length(net.nw_layers);

net.ind_n_layers = [net.n_in net.nw_hidden_layers net.n_out];
[net.W,net.b,vec_W,vec_b] = fun_initialize_param(net.ind_n_layers);

net.nW = numel(vec_W);
net.Wb = [vec_W; vec_b];

