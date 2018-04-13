function [f,g] = z_p_l_update(z_p_l,z_l_p,z_p_l_update_constant,matrix_feature)
%deal f

z_p_l=reshape(z_p_l, 1, matrix_feature);
z_p_l_sum = sum(z_p_l .* z_l_p);

f=(z_p_l_sum + z_p_l_update_constant) .^ 2;

%deal g

g=2 .* (z_p_l_sum + z_p_l_update_constant) .* z_l_p;
g=g(:);

legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'list_update_l_u_three: g is not legal!'
	end

end %endfunction