function net = fun_initialize_datasets(x_dataset,t_dataset,fac_train)

ind_training = 1:floor(fac_train*size(x_dataset,2));
% % ind_training = 1:floor(1*size(x_dataset,2));
ind_test = ind_training(end)+1:size(x_dataset,2);

m = size(x_dataset,2);
ind_random = randperm(m);

ind_training = ind_random(ind_training);
ind_test = ind_random(ind_test);

net.x_dataset_train = x_dataset(:,ind_training);
net.x_dataset_test  = x_dataset(:,ind_test);
net.t_dataset_train = t_dataset(:,ind_training);
net.t_dataset_test  = t_dataset(:,ind_test);
