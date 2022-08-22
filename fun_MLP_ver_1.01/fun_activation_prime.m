function x_k = fun_activation_prime(a_k,kind)

switch kind
    case 'logistic'
        x_k = fun_logistic_function_prime(a_k);
    case 'tanh'
        x_k = fun_tanh_prime(a_k);
    case 'relu'
        x_k = zeros(size(a_k));
        x_k(a_k > 0) = 1;
    case 'gelu'
        x = a_k;
        x_k = erf((2^(1/2)*x)/2)/2 + (2^(1/2)*x.*exp(-x.^2/2))/(2*pi^(1/2)) + 1/2;
    case 'sgelu'
        x = a_k;
        x_k = .1*erf((2^(1/2)*x)/2) + (2^(1/2)*x.*exp(-x.^2/2))/pi^(1/2);
    case 'SoftPlus'
        x = a_k;
        x_k = 1./(1+exp(-x));
    case 'BentIdentity'
        x = a_k;
        x_k = 1 + .5*x.*(sqrt(x.^2+1));
    case 'ISRLU'
        x = a_k;
        alpha = 1;
        x_k = ones(size(x));
        x_k(x >= 0) = x(x >= 0);
        x_k(x < 0) = (1./sqrt(1 + alpha*x(x < 0).^2)).^3;
end

end


function y = fun_logistic_function_prime(x)
y = exp(-x)./(1 + exp(-x)).^2;
end


function y = fun_tanh_prime(x)
y = 1 - tanh(x).^2;
end
























