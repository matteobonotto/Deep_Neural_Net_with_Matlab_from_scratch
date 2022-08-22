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

%% (R,Z) centroid

options_MLP.nw_hidden_layers = 50;
options_MLP.f_activation = 'tanh'; % Iteration  3900 | Cost: 1.068729e-05; error 1.0e-04*[0.7409    0.4863]
% % options_MLP.f_activation = 'logistic'; % Iteration  9400 | Cost: 9.976563e-06 ; error 1.0e-03*[0.4851    0.7681]
% % options_MLP.f_activation = 'relu'; % fails
% options_MLP.f_activation = 'gelu'; % Iteration  4000 | Cost: 1.016249e-05 ; error 1.0e-03*[-0.0857    0.1362]
% % options_MLP.f_activation = 'sgelu'; %  ; fails 

% % options_MLP.f_activation = 'SoftPlus'; % Iteration  4000 | Cost: 1.016249e-05 ; error 1.0e-03*[-0.0857    0.1362]
% % options_MLP.f_activation = 'BentIdentity'; % Iteration  4000 | Cost: 1.016249e-05 ; error 1.0e-03*[-0.0857    0.1362]
% % options_MLP.f_activation = 'ISRLU'; % Iteration  4000 | Cost: 1.016249e-05 ; error 1.0e-03*[-0.0857    0.1362]
options_MLP.threshold = 1e-5;
options_MLP.maxIter = 10000;

tic
net = fun_main_MLP(X_train,y_train,options_MLP);
toc
%%% predict

figure
semilogy(1:numel(net.costArray),net.costArray)

















