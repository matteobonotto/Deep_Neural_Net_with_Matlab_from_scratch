function net = fun_train_MLP(net)

%% Cost function for regression problem 
net.cf = @(Wb) fun_CostFunctionRegression(Wb,net);

 
%% Minimization (Conjugate Gradient)
options.MaxIter = net.maxIter;
options.threshold = net.threshold;
[net.Wb, net.costArray, net.iter] = fmincg(net.cf, net.Wb, options);


%% Minimization (Differential Evolution)
% % net.cf = @(Wb) fun_CostFunctionRegression_DE(Wb,net);
% % options.MaxIter = 5000;
% % options.threshold = 1e-10;
% % [net.Wb, net.costArray, net.iter] = fun_minimalDE_2(net.cf,net.Wb,options);

% % figure
% % semilogy(1:it,vec_fxbest)
% % hold on
% % semilogy(1:net.iter, net.costArray)


%% Minimization (quase-Newton)

% % options = optimoptions('fminunc', ...
% %     'Algorithm','trust-region', ...
% %     'SpecifyObjectiveGradient',true, ...
% %     'display', 'iter');
% % 
% % options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton', ...
% %     'UseParallel',true);
% % 
% % 
% % [net.Wb, net.costArray] = fminunc(net.cf,net.Wb,options);

