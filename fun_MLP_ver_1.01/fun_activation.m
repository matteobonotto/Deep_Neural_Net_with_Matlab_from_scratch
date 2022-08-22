function x_k = fun_activation(a_k,kind)

switch kind
    case 'logistic'
        x_k = fun_logistic_function(a_k);
    case 'tanh'
        x_k = tanh(a_k);
    case 'relu'
        x_k = zeros(size(a_k));
        x_k(a_k > 0) = a_k(a_k > 0);
    case 'gelu'
        x = a_k;
        x_k = (x.*(erf((2^(1/2)*x)/2) + 1))/2;
    case 'sgelu'
        x = a_k;
        x_k = .1*x.*erf((2^(1/2)*x)/2);
    case 'unit'
        x_k = a_k;
    case 'SoftPlus'
        x = a_k;
        x_k = log(1+exp(x));
    case 'BentIdentity'
        x = a_k;
        x_k = x + .5*(sqrt(x.^2+1)-1);
    case 'ISRLU'
        x = a_k;
        alpha = 1;
        x_k = zeros(size(x));
        x_k(x >= 0) = x(x >= 0);
        x_k(x < 0) = x(x < 0)./(1 + alpha*x(x < 0).^2);
end

end

% % syms x
% % gelu = .5*x.*(1 + erf(x/sqrt(2)))
% % gelu_prime = diff(gelu,x)
% % 
% % sgelu = x.*(erf(x/sqrt(2)))
% % sgelu_prime = diff(sgelu,x)
% % 
% % 
% % figure
% % fplot(gelu); hold on;
% % fplot(gelu_prime)
% % fplot(sgelu)
% % fplot(sgelu_prime)

function y = fun_logistic_function(x)
y = 1./(1 + exp(-x));
end


