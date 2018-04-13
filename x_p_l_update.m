function [f,g] = x_p_l_update(x_p_l,x_l_p,x_p_l_update_constant,matrix_feature)
%deal f

x_p_l=reshape(x_p_l, 1, matrix_feature);
x_p_l_sum = sum(x_p_l .* x_l_p);

f=(x_p_l_sum + x_p_l_update_constant) .^ 2;

%deal g

g=2 .* (x_p_l_sum + x_p_l_update_constant) .* x_l_p;
g=g(:);

legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'list_update_l_u_three: g is not legal!'
	end

end %endfunction