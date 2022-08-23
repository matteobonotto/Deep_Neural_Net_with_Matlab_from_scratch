function [W,b,vec_W,vec_b] = fun_initialize_param(ind_n_layers)

n_in = ind_n_layers(1);
nw_layers = ind_n_layers(2:end);
%%% Initialize W and b
n_layers = length(nw_layers);

% % W{n_layers} = {};
% % b{n_layers} = {};
% % for ii = 1:n_layers
% %     if ii == 1
% %         n_rows = nw_layers(ii);
% %         n_cols = n_in;
% %     else
% %         n_rows = nw_layers(ii);
% %         n_cols = nw_layers(ii-1);
% %     end
% %     W{ii} = rand(n_rows,n_cols)/(n_rows*n_cols);
% %     b{ii} = rand(n_rows,1)/n_rows;
% %     
% %     % %     [W_ii,b_ii] = fun_featureNormalize_training(rand(n_rows,n_cols),rand(n_rows,1));
% %     % %
% %     % %     W{ii} = W_ii.data;
% %     % %     b{ii} = b_ii.data;
% %     
% %     
% % end
% % 
% % %%% Vectorize
% % vec_W = fun_Cell2Vector(W);
% % vec_b = fun_Cell2Vector(b);


%%
layers = ind_n_layers;

nel = bsxfun(@times, layers(1:end-1), layers(2:end));
% function for determining the amplitude of init values for each layer
efun = @(x, y) sqrt(6)./(sqrt(x + y));
% the init apmlitudes for each layer
epsilon_init = repelem(bsxfun(efun, layers(1:end-1), layers(2:end)), nel);
% the init weights for each neuron
vec_W = (2*rand(sum(nel), 1) - 1).*epsilon_init'*.1;
W = fun_Vector2CellMat(vec_W,ind_n_layers);


nel = bsxfun(@times, layers(2:end), 1);
% function for determining the amplitude of init values for each layer
efun = @(x, y) sqrt(6)./(sqrt(x + y));
% the init apmlitudes for each layer
epsilon_init = repelem(bsxfun(efun, layers(1:end-1), layers(2:end)), nel);
% the init weights for each neuron
vec_b = (2*rand(sum(nel), 1) - 1).*epsilon_init'*.1;
b = fun_Vector2CellVec(vec_b,ind_n_layers);





