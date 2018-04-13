function [f,g] = z_l_u_update(z_l_u,z_u_l,z_u_l_update_constant,matrix_feature)
%deal f

z_l_u=reshape(z_l_u, 1, matrix_feature);
z_u_l_sum = sum(z_u_l .* z_l_u);

f=(z_u_l_sum + z_u_l_update_constant) .^ 2;

%deal g

g=2 .* (z_u_l_sum + z_u_l_update_constant) .* z_u_l;
g=g(:);

legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'list_update_l_u_three: g is not legal!'
	end

end %endfunction