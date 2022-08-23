function net = fun_initialize_datasets(x_dataset,t_dataset,options_MLP)

% Train/test split
[x_train, x_test, t_train, t_test] = fun_train_test_split(x_dataset,t_dataset,options_MLP.train_test_split_ratio);

net.x_dataset_train = x_train;
net.x_dataset_test  = x_test;
net.t_dataset_train = t_train;
net.t_dataset_test  = t_test;