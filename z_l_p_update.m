function [f,g] = z_l_p_update(z_l_p,z_p_l,z_p_l_update_constant,matrix_feature)
%deal f

z_l_p=reshape(z_l_p, 1, matrix_feature);
z_p_l_sum = sum(z_p_l .* z_l_p);

f=(z_p_l_sum + z_p_l_update_constant) .^ 2;

%deal g

g=2 .* (z_p_l_sum + z_p_l_update_constant) .* z_p_l;
g=g(:);

legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'list_update_l_u_three: g is not legal!'
	end

end %endfunction