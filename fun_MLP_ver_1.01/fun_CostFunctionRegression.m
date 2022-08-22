function [f_CF,grad_CF] = fun_CostFunctionRegression(Wb,net)


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

% % h_nn = z{end};
% % J_matrix = -((t_dataset+1).*log((h_nn + 1)/2)/2 + (1 - (t_dataset+1)/2).*log(1 - (h_nn+1)/2));
% % J_matrix(isnan(J_matrix)) = 0;
% % f_CF = sum(J_matrix(:));
% % 
% % % normalizing
% % f_CF = f_CF/m;


% % [out,z,a] = fun_predict_MLP(net,net.x_dataset_train);
% % En = .5*(out - net.t_dataset_train).^2;
% % f_CF = sum(En(:))/m;



%% Backpropagation

delta{n_layers} = {};
Delta_W{n_layers} = {};
Delta_b{n_layers} = {};

% last layer (l = L = 2)
delta_end = (z{end} - t_dataset);
Delta_end_w = delta_end*z{end-1}'/m;
Delta_end_b = sum(delta_end,2)/m;

delta{n_layers} = delta_end;
Delta_W{n_layers} = Delta_end_w;
Delta_b{n_layers} = Delta_end_b;

for ii = n_layers-1:-1:1
    delta{ii} = (W{ii+1}.'*delta{ii+1}).*fun_activation_prime(a{ii},f_activation);
    Delta_W{ii} = delta{ii}*z{(ii-1)+1}.'/m; % here z{} stores also the input value, thus (ii-1)+1
    Delta_b{ii} = sum(delta{ii},2)/m;
end


%%% store Delta on a single vector
grad_CF = [fun_Cell2Vector(Delta_W); ...
    fun_Cell2Vector(Delta_b)];
























