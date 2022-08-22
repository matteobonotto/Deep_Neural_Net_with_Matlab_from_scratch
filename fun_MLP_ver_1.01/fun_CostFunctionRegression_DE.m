function [f_CF] = fun_CostFunctionRegression_DE(Wb,net)


nW           = net.nW;
ind_n_layers = net.ind_n_layers;
x_dataset    = net.X.data;
t_dataset    = net.T.data;
f_activation = net.f_activation;


%%

W = Wb(1:nW);
b = Wb(nW+1:end);

W = fun_Vector2CellMat(W,ind_n_layers);
b = fun_Vector2CellVec(b,ind_n_layers);

n_layers = length(W);
m = size(t_dataset, 2); % Numeber of examples


%% Compute error of cost function

%%% Forward Propagation
z{n_layers+1} = {};
a{n_layers} = {};

z{1} = x_dataset;

for ii = 1:n_layers
    a{ii} = b{ii} + W{ii}*z{ii};
    if ii == n_layers; f_activation_ii = 'unit'; else; f_activation_ii = f_activation; end
    z{ii+1} = fun_activation(a{ii},f_activation_ii);
end


%%% Error of CF
En = .5*(z{end} - t_dataset).^2;
f_CF = sum(En(:))/m;























