function vec_f = fun_vec_cf(cf,x)
vec_f = zeros(size(x,1),1);
for ii = 1:size(x,1)
    vec_f(ii) = cf(x(ii,:).');
end