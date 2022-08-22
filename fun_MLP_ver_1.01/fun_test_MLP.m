function net = fun_test_MLP(net)

%%% Train dataset
out = fun_predict_MLP(net,net.x_dataset_train);
En = .5*(out - net.t_dataset_train).^2;
net.Error_train = sum(En(:))/size(net.t_dataset_train,2);


%%% Test dataset
out = fun_predict_MLP(net,net.x_dataset_test);
En = .5*(out - net.t_dataset_test).^2;
net.Error_test = sum(En(:))/size(net.t_dataset_test,2);

