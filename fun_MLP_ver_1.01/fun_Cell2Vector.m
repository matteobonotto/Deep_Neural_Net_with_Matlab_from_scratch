function vec_element = fun_Cell2Vector(cell_element)

n_layers = length(cell_element);

No_of_elements = cellfun(@numel, cell_element);
vec_element = zeros(sum(No_of_elements),1);

ind_start = 1;
for ii = 1:n_layers
    ind_end = ind_start+No_of_elements(ii)-1;
    vec_element(ind_start:ind_end) = cell_element{ii}(:);
    ind_start = ind_end+1;
end
