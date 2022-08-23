function [x_train, x_test, t_train, t_test] = fun_train_test_split(x_dataset,t_dataset,factor)

ind_training = 1:floor(factor*size(x_dataset,2));
% % ind_training = 1:floor(1*size(x_dataset,2));
ind_test = ind_training(end)+1:size(x_dataset,2);

m = size(x_dataset,2);
ind_random = randperm(m);

ind_training = ind_random(ind_training);
ind_test = ind_random(ind_test);

x_train = x_dataset(:,ind_training);
x_test  = x_dataset(:,ind_test);
t_train = t_dataset(:,ind_training);
t_test  = t_dataset(:,ind_test);
