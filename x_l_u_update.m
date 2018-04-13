function [f,g] = x_l_u_update(x_l_u,x_u_l,x_u_l_update_constant,matrix_feature)
%deal f

x_l_u=reshape(x_l_u, 1, matrix_feature);
x_u_l_sum = sum(x_u_l .* x_l_u);

f=(x_u_l_sum + x_u_l_update_constant) .^ 2;

%deal g

g=2 .* (x_u_l_sum + x_u_l_update_constant) .* x_u_l;
g=g(:);

legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'list_update_l_u_three: g is not legal!'
	end

end %endfunction