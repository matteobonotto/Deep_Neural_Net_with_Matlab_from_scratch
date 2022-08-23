%% 
clc; close all; clearvars;

restoredefaultpath

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

addpath ./fun_MLP_ver_1.01

%%
dataset = load('housing_data.txt');

X_train = dataset(:,1:end-1).';
y_train = dataset(:,end).';

options.Normalize_input = true;
options.Normalize_output = true;
[X,T] = fun_featureNormalize_training(X_train,y_train,options);
X_train = X.data;
y_train = T.data;

% % [X_train, X_test, y_train, y_test] = ...
% %     fun_train_test_split(X_train,y_train,.8);


%% Train network
options_MLP.nw_hidden_layers = [50];
options_MLP.f_activation = 'tanh'; 
options_MLP.threshold = 1e-3;
options_MLP.maxIter = 1000;
options_MLP.train_test_split_ratio = .8;

options_MLP.Normalize_input = false;
options_MLP.Normalize_output = false;

tic
net = fun_main_MLP(X_train,y_train,options_MLP);
toc
%%% predict

figure    
semilogy(1:numel(net.costArray),net.costArray)
grid on; hold on;
xlabel('Iterations')
ylabel('Cost')

figure
plot(net.t_dataset_train, fun_predict_MLP(net,net.x_dataset_train), 'o')
axis equal

fprintf('The R^2 for is: %f\n', ...
    rsquare(fun_predict_MLP(net,net.x_dataset_train),net.t_dataset_train))














