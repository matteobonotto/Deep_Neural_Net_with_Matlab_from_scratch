function [out,z,a] = fun_predict_MLP(net,x_test)


%% normalize test vector
X_test = fun_featureNormalize_test(net,x_test);


%%
Wb           = net.Wb;
nW           = net.nW;
ind_n_layers = net.ind_n_layers;
t_dataset    = net.T.data;
f_activation = net.f_activation;

W = Wb(1:nW);
b = Wb(nW+1:end);

W = fun_Vector2CellMat(W,ind_n_layers);
b = fun_Vector2CellVec(b,ind_n_layers);

n_layers = length(W);
m = size(t_dataset, 2); % Numeber of examples


%% Forward Propagation
z{n_layers+1} = {};
a{n_layers} = {};

z{1} = X_test;

for ii = 1:n_layers
    a{ii} = b{ii} + W{ii}*z{ii};
    if ii == n_layers; f_activation_ii = 'unit'; else; f_activation_ii = f_activation; end
    z{ii+1} = fun_activation(a{ii},f_activation_ii);
end

out = z{end};


%% De-normalize output
out = fun_denormalize_output(out,net);























