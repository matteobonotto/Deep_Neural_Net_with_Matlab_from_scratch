function cell_element = fun_Vector2CellVec(vec_element,ind_Vec2Cell)


cell_element{length(ind_Vec2Cell)-1} = {};
offset = 0;
for ii = 1:length(ind_Vec2Cell)-1
    ind_loc = 1:ind_Vec2Cell(ii+1);
    ind_glob = ind_loc + offset;
    cell_element{ii} = vec_element(ind_glob);
    offset = ind_glob(end);
end