function net = fun_main_MLP(x_dataset,t_dataset,options_MLP)

%% Split data in trainind and test sets
net = fun_initialize_datasets(x_dataset,t_dataset,options_MLP);

%% Initialize network
net = fun_initialize_MLP(net,options_MLP);

%% Train network
net = fun_train_MLP(net);

%% Test network
net = fun_test_MLP(net);


