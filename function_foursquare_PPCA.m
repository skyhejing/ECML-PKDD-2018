addpath(genpath(pwd));
%setting for minFunc
options = [];
options.display = 'none';
options.Method = 'lbfgs';
options.maxFunEvals = 3;
%end(setting for minFunc)

lamda=1;    %init the lamda
matrix_feature=8;    %the number of matrix feature


%transfer the distance and time with nice metric

%init tensor parameter
x_u_l=normrnd(0,1/lamda,[size(user_unique,1),matrix_feature]);
x_l_u=normrnd(0,1/lamda,[size(location_unique,1),matrix_feature]);
x_p_l=normrnd(0,1/lamda,[size(location_unique,1),matrix_feature]);
x_l_p=normrnd(0,1/lamda,[size(location_unique,1),matrix_feature]);

z_u_l=normrnd(0.01,0.001,[size(user_unique,1),matrix_feature]);
z_l_u=normrnd(0.01,0.001,[size(location_unique,1),matrix_feature]);
z_p_l=normrnd(0.01,0.001,[size(location_unique,1),matrix_feature]);
z_l_p=normrnd(0.01,0.001,[size(location_unique,1),matrix_feature]);

%init personalized rho
rho=normrnd(150,1/lamda,1);

%init w
w=normrnd(0,1/lamda,size(user_unique,1),1);

%init sigma_1
sigma_1_square=normrnd(0,1/lamda,size(user_unique,1),1);

%init sigma_2
sigma_2_square=normrnd(0,1/lamda,size(user_unique,1),1);

[row_trainset, col_trainset]=size(trainset);
[row_testset, col_testset]=size(testset);

%init the iteration count
iteration_i=1;

%init the iteration of M step
iteration_M_tensor=2000;
iteration_M_user=200;
iteration_M_rho=5;

%the metric of distance
distance_metric_meter=1;
distance_metric=1 ./ ((trainset(:,8) * distance_metric_meter)+ (0.5 * distance_metric_meter));


%the metric of time
time_metric=3600;
testset(:,22) = testset(:,22) ./ time_metric;

%init result.txt
fid=fopen('result.txt','a+');
%end(init result.txt)

%count distance from the testset
distance_all=testset(:,8);
Mylength(1)= length(find(distance_all>=0&distance_all<5));
Mylength(2)= length(find(distance_all>=5&distance_all<10));
Mylength(3)= length(find(distance_all>=10&distance_all<20));
Mylength(4)= length(find(distance_all>=20&distance_all<50));
Mylength(5)= length(find(distance_all>=50&distance_all<100));
Mylength(6)= length(find(distance_all>=100&distance_all<200));
Mylength(7)= length(find(distance_all>=200&distance_all<500));
Mylength(8)= length(find(distance_all>=500&distance_all<800));
Mylength_cum=cumsum(Mylength);
%end(count distance from the testset)

%count time from the testset
time_all=testset(:,22);
Mytime(1)= length(find(time_all>=0&time_all<(900/time_metric)));
Mytime(2)= length(find(time_all>=(900/time_metric) &time_all< (1800/time_metric) ));
Mytime(3)= length(find(time_all>=(1800/time_metric) &time_all< (3600/time_metric) ));
Mytime(4)= length(find(time_all>=(3600/time_metric) &time_all< (21600/time_metric) ));
Mytime(5)= length(find(time_all>= (21600/time_metric) &time_all< (43200/time_metric) ));
Mytime(6)= length(find(time_all>= (43200/time_metric) &time_all< (86400/time_metric) ));
Mytime(7)= length(find(time_all>= (86400/time_metric) &time_all< (172800/time_metric) ));
Mytime(8)= length(find(time_all>= (172800/time_metric) &time_all< (604800/time_metric) ));
Mytime_cum=cumsum(Mytime);
%end(count time from the testset)
while 1
    iteration_i=iteration_i+1;
    
    %E step
    
    %deal with x_tensor
    x_u_l_trainset=x_u_l(trainset(:,1),:);
    x_l_u_trainset=x_l_u(trainset(:,3),:);
    x_l_p_trainset=x_l_p(trainset(:,3),:);
    x_p_l_trainset=x_p_l(trainset(:,2),:);
    x_rho=rho;
    
    x_u_l_sum=sum(x_u_l_trainset .* x_l_u_trainset,2);
    x_p_l_sum=sum(x_l_p_trainset .* x_p_l_trainset,2);
    distance_rho=x_rho .* distance_metric;
    
    x_tenosr=x_u_l_sum + x_p_l_sum+distance_rho;
    %end (deal with x_tensor)
    
    
    %deal with z_tensor
    z_u_l_trainset=z_u_l(trainset(:,1),:);
    z_l_u_trainset=z_l_u(trainset(:,3),:);
    z_l_p_trainset=z_l_p(trainset(:,3),:);
    z_p_l_trainset=z_p_l(trainset(:,2),:);
    
    z_u_l_sum=sum(z_u_l_trainset .* z_l_u_trainset,2);
    z_p_l_sum=sum(z_l_p_trainset .* z_p_l_trainset,2);
    
    z_tenosr=z_u_l_sum + z_p_l_sum;
    %end(deal with z_tensor)
    
    %calcute the E(z)
    sigma_1_square_trainset=sigma_1_square(trainset(:,1),:);
    
    sigma_2_square_trainset=sigma_2_square(trainset(:,1),:);
    
    w_trainset=w(trainset(:,1),:);
    w_square_trainset=w_trainset .^ 2;
    
    x_minus_x_tensor= trainset(:,23) - x_tenosr;
    
    E_z_num_first=sigma_1_square_trainset .* w_trainset .* x_minus_x_tensor;
    E_z_num_second=z_tenosr .* sigma_2_square_trainset;
    E_z_num=E_z_num_first +E_z_num_second;
    
    E_z_den= sigma_2_square_trainset + w_square_trainset .* sigma_1_square_trainset;
    E_z= E_z_num ./ E_z_den;
    %end(calcute the E(z))
    
    %calcute the E(z_square)
    E_z_square_first_num=sigma_1_square_trainset .* sigma_2_square_trainset;
    E_z_square_first= E_z_square_first_num ./ E_z_den;
    
    E_z_square= E_z_square_first + E_z .^ 2;
    
    %end(calcute the E(z_square))
    %end(E step)
    
    %M step
    for i_two=1:iteration_M_tensor
    rand_trainset_number=randsample(row_trainset,1);
    line_numbers=trainset(trainset(rand_trainset_number,1)==trainset(:,1),24);
    userID_rand=trainset(rand_trainset_number,1);
    
    %update x_u_l
    x_u_l_rand=x_u_l(userID_rand,:);
    x_l_u_rand=x_l_u(trainset(rand_trainset_number,3),:);
    x_l_p_rand=x_l_p(trainset(rand_trainset_number,3),:);
    x_p_l_rand=x_p_l(trainset(rand_trainset_number,2),:);
    distance_metric_sample=distance_metric(rand_trainset_number);
    rho_sample=rho;
    
    x_u_l_update_constant = sum(x_l_p_rand .* x_p_l_rand) + (rho_sample .*distance_metric_sample)- trainset(rand_trainset_number,23)  + (w(userID_rand) .* E_z(rand_trainset_number));
    opt_x_u_l = minFunc(@x_u_l_update,x_u_l_rand(:),options,x_l_u_rand,x_u_l_update_constant,matrix_feature);
    x_u_l(userID_rand,:)=reshape(opt_x_u_l,1, matrix_feature);
    %end(update x_u_l)
    
    %update x_l_u
    opt_x_l_u = minFunc(@x_l_u_update,x_l_u_rand(:),options,x_u_l_rand,x_u_l_update_constant,matrix_feature);
    x_l_u(trainset(rand_trainset_number,3),:)=reshape(opt_x_l_u,1, matrix_feature);
    %end(update x_l_u)
    
    %update x_l_p
    x_p_l_update_constant = sum(x_l_u_rand .* x_u_l_rand) + (rho_sample .*distance_metric_sample)- trainset(rand_trainset_number,23)  + (w(userID_rand) .* E_z(rand_trainset_number));
    opt_x_l_p = minFunc(@x_l_p_update,x_l_p_rand(:),options,x_p_l_rand,x_p_l_update_constant,matrix_feature);
    x_l_p(trainset(rand_trainset_number,3),:)=reshape(opt_x_l_p,1, matrix_feature);
    %end(update x_l_p)
    
    %update x_p_l
    opt_x_p_l = minFunc(@x_p_l_update,x_p_l_rand(:),options,x_l_p_rand,x_p_l_update_constant,matrix_feature);
    x_p_l(trainset(rand_trainset_number,2),:)=reshape(opt_x_p_l,1, matrix_feature);
    %end(update x_p_l)
    
    %update z_u_l
    z_u_l_rand=z_u_l(userID_rand,:);
    z_l_u_rand=z_l_u(trainset(rand_trainset_number,3),:);
    z_l_p_rand=z_l_p(trainset(rand_trainset_number,3),:);
    z_p_l_rand=z_p_l(trainset(rand_trainset_number,2),:);
    
    z_u_l_update_constant = sum(z_l_p_rand .* z_p_l_rand) - E_z(rand_trainset_number);
    opt_z_u_l = minFunc(@z_u_l_update,z_u_l_rand(:),options,z_l_u_rand,z_u_l_update_constant,matrix_feature);
    z_u_l(userID_rand,:)=reshape(opt_z_u_l,1, matrix_feature);
    %end(update z_u_l)
    
    %update z_l_u
    opt_z_l_u = minFunc(@z_l_u_update,z_l_u_rand(:),options,z_u_l_rand,z_u_l_update_constant,matrix_feature);
    z_l_u(trainset(rand_trainset_number,3),:)=reshape(opt_z_l_u,1, matrix_feature);
    %end(update x_l_u)
    
    %update z_l_p
    z_p_l_update_constant = sum(z_l_u_rand .* z_u_l_rand) - E_z(rand_trainset_number);
    opt_z_l_p = minFunc(@z_l_p_update,z_l_p_rand(:),options,z_p_l_rand,z_p_l_update_constant,matrix_feature);
    z_l_p(trainset(rand_trainset_number,3),:)=reshape(opt_z_l_p,1, matrix_feature);
    %end(update x_l_p)
    
    %update z_p_l
    opt_z_p_l = minFunc(@z_p_l_update,z_p_l_rand(:),options,z_l_p_rand,z_p_l_update_constant,matrix_feature);
    z_p_l(trainset(rand_trainset_number,2),:)=reshape(opt_z_p_l,1, matrix_feature);
    %end(update x_p_l)
    
    end % for i_two=1:iteration_M_tensor
    
    for i_two=1:iteration_M_user
    %update sigma_1_square
    E_z_rand=E_z(line_numbers);
    E_z_square_rand=E_z_square(line_numbers);
    
    z_tenosr_rand=z_tenosr(line_numbers);
    sigma_1_square_update_second=2 .* E_z_rand .* z_tenosr_rand;
    sigma_1_square_update_third=z_tenosr_rand .^ 2;
    sigma_1_square_update_sum=sum(E_z_square_rand - sigma_1_square_update_second + sigma_1_square_update_third);
    
    sigma_1_square(userID_rand) = sigma_1_square_update_sum /size(line_numbers,1);
    %end(update sigma_1_square)
    
    %update sigma_2_square
    x_rand=trainset(line_numbers,23);
    sigma_2_square_update_1=x_rand .^ 2;
    
    w_rand=w(trainset(rand_trainset_number,1));
    sigma_2_square_update_2=(w_rand .^ 2) .* E_z_square_rand;
    
    x_tenosr_rand=x_tenosr(line_numbers);
    sigma_2_square_update_3=x_tenosr_rand .^ 2;
    
    sigma_2_square_update_4= 2 .* x_rand .* w_rand .* E_z_rand;
    
    sigma_2_square_update_5= 2 .* x_rand .* x_tenosr_rand;
    
    sigma_2_square_update_6 = 2 .* w_rand .* E_z_rand .* x_tenosr_rand;
    
    sigma_2_square_update_sum= sum(sigma_2_square_update_1 + sigma_2_square_update_2 + sigma_2_square_update_3 - sigma_2_square_update_4 -sigma_2_square_update_5 + sigma_2_square_update_6);
    sigma_2_square(userID_rand) = sigma_2_square_update_sum / size(line_numbers,1);
    %end(update sigma_2_square)
    
    %update w
    x_minus_x_tensor_rand=x_minus_x_tensor(line_numbers);
    w_update_num=sum(E_z_rand .* x_minus_x_tensor_rand);
    
    w_update_den=sum(E_z_square_rand);
    w(userID_rand)=w_update_num/w_update_den;
    %end(update w)
    
    end %for i_two=1:iteration_M_user
    
    for i_two=1:iteration_M_rho
    %update rho
    w_rand_rho=w(trainset(:,1));
    rho_update_num_first_1=w_rand_rho .* E_z;
    x_rand_rho=trainset(:,23);
    rho_update_num_first_2=x_rand_rho-rho_update_num_first_1;
    
    rho_update_num_first_sum= sum(distance_metric .* rho_update_num_first_2);
    
    
    rho_update_num_second_1= x_u_l_sum .* distance_metric;
    rho_update_num_second_2= x_p_l_sum .* distance_metric;
    rho_update_num_second_sum = sum(rho_update_num_second_1 + rho_update_num_second_2);
    
    distance_metric_square=distance_metric .^ 2;
    rho_update_den=sum(distance_metric_square);
    
    rho = (rho_update_num_first_sum - rho_update_num_second_sum) / rho_update_den;
    disp(['rho = ' num2str(rho)]);
    %end(update rho)
    
    end %for i_two=1:iteration_M_rho

    %end(M step)
    
    %prediction
%     prediction_foursquare_PPCA( testset,x_l_u,x_u_l,x_l_p,x_p_l,z_l_u,z_u_l,z_l_p,z_p_l,rho,w,location_unique_test,distance_matrix_frac,time_metric,distance_metric_meter,2)
    [Myresult_1,Myresult_5,Myresult_10,Myresult_20,Myresult_30,Myresult_40,Myresult_50,right_distance,right_when,Myresult_time_quarter,Myresult_time_half,Myresult_time_1,Myresult_time_6,Myresult_time_12,Myresult_time_24,Myresult_time_48,Myresult_time_168,Myresult_time_location_1,Myresult_time_location_5,Myresult_time_location_10,Myresult_time_location_20,Myresult_time_location_50,Myresult_location10_time_1,Myresult_location10_time_6,Myresult_location10_time_12,Myresult_location10_time_24,Myresult_location10_time_48,Myresult_location10_time_168,right_i_1,right_i_5,right_i_10,right_i_20,right_i_50,time_distance_MAE,time_distance_MAPE,top10_time_1_distance,top10_time_1_when,top20_time_1_when]=arrayfun(@(i) prediction_foursquare_PPCA( testset,x_l_u,x_u_l,x_l_p,x_p_l,z_l_u,z_u_l,z_l_p,z_p_l,rho,w,location_unique_test,distance_matrix_frac,time_metric,distance_metric_meter,i),1:row_testset);
    %end(prediction)
    
    %the precision of location recommendation
    disp(['top1 = ' num2str(sum(Myresult_1)./ row_testset)]);
    disp(['top5 = ' num2str(sum(Myresult_5)./ row_testset)]);
    disp(['top10 = ' num2str(sum(Myresult_10)./ row_testset)]);
    disp(['top20 = ' num2str(sum(Myresult_20)./ row_testset)]);
    disp(['top30 = ' num2str(sum(Myresult_30)./ row_testset)]);
    disp(['top40 = ' num2str(sum(Myresult_40)./ row_testset)]);
    disp(['top50 = ' num2str(sum(Myresult_50)./ row_testset)]);
    
    fprintf(fid,'rho = %s\n',num2str(rho));
    fprintf(fid,'top1 = %s\n',num2str(sum(Myresult_1)./ row_testset));
    fprintf(fid,'top5 = %s\n',num2str(sum(Myresult_5)./ row_testset));
    fprintf(fid,'top10 = %s\n',num2str(sum(Myresult_10)./ row_testset));
    fprintf(fid,'top20 = %s\n',num2str(sum(Myresult_20)./ row_testset));
    fprintf(fid,'top30 = %s\n',num2str(sum(Myresult_30)./ row_testset));
    fprintf(fid,'top40 = %s\n',num2str(sum(Myresult_40)./ row_testset));
    fprintf(fid,'top50 =%s\n',num2str(sum(Myresult_50)./ row_testset));
    fprintf(fid,'\n');
    %end(the precision of location recommendation)
    
    %the precision of time recommendation
    disp(['MAE = ' num2str(sum(time_distance_MAE)./ row_testset)]);
    disp(['MAPE = ' num2str(sum(time_distance_MAPE)./ row_testset)]);
    disp(['hour_quarter = ' num2str(sum(Myresult_time_quarter)./ row_testset)]);
    disp(['hour_half = ' num2str(sum(Myresult_time_half)./ row_testset)]);
    disp(['hour1 = ' num2str(sum(Myresult_time_1)./ row_testset)]);
    disp(['hour6 = ' num2str(sum(Myresult_time_6)./ row_testset)]);
    disp(['hour12 = ' num2str(sum(Myresult_time_12)./ row_testset)]);
    disp(['hour24 = ' num2str(sum(Myresult_time_24)./ row_testset)]);
    disp(['hour48 = ' num2str(sum(Myresult_time_48)./ row_testset)]);
    disp(['hour168 = ' num2str(sum(Myresult_time_168)./ row_testset)]);
    disp(['rho = ' num2str(rho)]);
    disp(['iteration = ' num2str(iteration_i)]);
    
    fprintf(fid,'MAE = %s\n',num2str(sum(time_distance_MAE)./ row_testset));
    fprintf(fid,'MAPE = %s\n',num2str(sum(time_distance_MAPE)./ row_testset));
    fprintf(fid,'hour_quarter = %s\n',num2str(sum(Myresult_time_quarter)./ row_testset));
    fprintf(fid,'hour_half = %s\n',num2str(sum(Myresult_time_half)./ row_testset));
    fprintf(fid,'hour1 = %s\n',num2str(sum(Myresult_time_1)./ row_testset));
    fprintf(fid,'hour6 = %s\n',num2str(sum(Myresult_time_6)./ row_testset));
    fprintf(fid,'hour12 = %s\n',num2str(sum(Myresult_time_12)./ row_testset));
    fprintf(fid,'hour24 = %s\n',num2str(sum(Myresult_time_24)./ row_testset));
    fprintf(fid,'hour48 = %s\n',num2str(sum(Myresult_time_48)./ row_testset));
    fprintf(fid,'hour168 = %s\n',num2str(sum(Myresult_time_168)./ row_testset));
    fprintf(fid,'\n');
    %end(the precision of time recommendation)
    
    %the precision of location recommendation when the abs(time)=24hour
    fprintf(fid,'top1_hour24 = %s\n',num2str(sum(Myresult_time_location_1)./ row_testset));
    fprintf(fid,'top5_hour24 = %s\n',num2str(sum(Myresult_time_location_5)./ row_testset));
    fprintf(fid,'top10_hour24 = %s\n',num2str(sum(Myresult_time_location_10)./ row_testset));
    fprintf(fid,'top20_hour24 = %s\n',num2str(sum(Myresult_time_location_20)./ row_testset));
    fprintf(fid,'top50_hour24 =%s\n',num2str(sum(Myresult_time_location_50)./ row_testset));
    fprintf(fid,'\n');
    %end(the precision of location recommendation when the abs(time)=24hour)
    
    %the precison of time recommendation when top10 is right
    fprintf(fid,'hour1_top10 = %s\n',num2str(sum(Myresult_location10_time_1)./ row_testset));
    fprintf(fid,'hour6_top10 = %s\n',num2str(sum(Myresult_location10_time_6)./ row_testset));
    fprintf(fid,'hour12_top10 = %s\n',num2str(sum(Myresult_location10_time_12)./ row_testset));
    fprintf(fid,'hour24_top10 = %s\n',num2str(sum(Myresult_location10_time_24)./ row_testset));
    fprintf(fid,'hour48_top10 = %s\n',num2str(sum(Myresult_location10_time_48)./ row_testset));
    fprintf(fid,'hour168_top10 = %s\n',num2str(sum(Myresult_location10_time_168)./ row_testset));
    fprintf(fid,'\n');
    %end(the precison of time recommendation when top10 is right)
   
    
    %distance quantitative
    right_length(1)= length(find(right_distance>=0&right_distance<5));
    right_length(2)= length(find(right_distance>=5&right_distance<10));
    right_length(3)= length(find(right_distance>=10&right_distance<20));
    right_length(4)= length(find(right_distance>=20&right_distance<50));
    right_length(5)= length(find(right_distance>=50&right_distance<100));
    right_length(6)= length(find(right_distance>=100&right_distance<200));
    right_length(7)= length(find(right_distance>=200&right_distance<500));
    right_length(8)= length(find(right_distance>=500&right_distance<800));
    right_length_cum=cumsum(right_length);
    
%     disp(['distance5 = ' num2str(right_length(1)./ Mylength(1))]);
%     disp(['distance10 = ' num2str(right_length(2)./ Mylength(2))]);
%     disp(['distance20 = ' num2str(right_length(3)./ Mylength(3))]);
%     disp(['distance50 = ' num2str(right_length(4)./ Mylength(4))]);
%     disp(['distance100 =' num2str(right_length(5)./ Mylength(5))]);
%     disp(['distance200 = ' num2str(right_length(6)./ Mylength(6))]);
%     disp(['distance500 = ' num2str(right_length(7)./ Mylength(7))]);
%     disp(['distance800 = ' num2str(right_length(8)./ Mylength(8))]);
    
    fprintf(fid,'distance5 = %s\n' ,num2str(right_length(1)./ Mylength(1)));
    fprintf(fid,'distance10 = %s\n' ,num2str(right_length(2)./ Mylength(2)));
    fprintf(fid,'distance20 = %s\n' ,num2str(right_length(3)./ Mylength(3)));
    fprintf(fid,'distance50 = %s\n' ,num2str(right_length(4)./ Mylength(4)));
    fprintf(fid,'distance100 =%s\n' ,num2str(right_length(5)./ Mylength(5)));
    fprintf(fid,'distance200 = %s\n' ,num2str(right_length(6)./ Mylength(6)));
    fprintf(fid,'distance500 = %s\n' ,num2str(right_length(7)./ Mylength(7)));
    fprintf(fid,'distance800 = %s\n' ,num2str(right_length(8)./ Mylength(8)));
    fprintf(fid,'\n');
    
    distance_quantitative=right_length_cum ./ Mylength_cum;
    fprintf(fid,'distance_quantitative5 = %s\n' ,num2str(distance_quantitative(1)));
    fprintf(fid,'distance_quantitative10 = %s\n' ,num2str(distance_quantitative(2)));
    fprintf(fid,'distance_quantitative20 = %s\n' ,num2str(distance_quantitative(3)));
    fprintf(fid,'distance_quantitative50 = %s\n' ,num2str(distance_quantitative(4)));
    fprintf(fid,'distance_quantitative100 =%s\n' ,num2str(distance_quantitative(5)));
    fprintf(fid,'distance_quantitative200 = %s\n' ,num2str(distance_quantitative(6)));
    fprintf(fid,'distance_quantitative500 = %s\n' ,num2str(distance_quantitative(7)));
    fprintf(fid,'distance_quantitative800 = %s\n' ,num2str(distance_quantitative(8)));
    fprintf(fid,'\n');
    %end(distance quantitative)
    
    %top10_time_1_distance_quantitative
    top10_time_1_length(1)= length(find(top10_time_1_distance>=0&top10_time_1_distance<5));
    top10_time_1_length(2)= length(find(top10_time_1_distance>=5&top10_time_1_distance<10));
    top10_time_1_length(3)= length(find(top10_time_1_distance>=10&top10_time_1_distance<20));
    top10_time_1_length(4)= length(find(top10_time_1_distance>=20&top10_time_1_distance<50));
    top10_time_1_length(5)= length(find(top10_time_1_distance>=50&top10_time_1_distance<100));
    top10_time_1_length(6)= length(find(top10_time_1_distance>=100&top10_time_1_distance<200));
    top10_time_1_length(7)= length(find(top10_time_1_distance>=200&top10_time_1_distance<500));
    top10_time_1_length(8)= length(find(top10_time_1_distance>=500&top10_time_1_distance<800));
    top10_time_1_length_cum=cumsum(top10_time_1_length);
    
    fprintf(fid,'distance5 = %s\n' ,num2str(top10_time_1_length(1)./ Mylength(1)));
    fprintf(fid,'distance10 = %s\n' ,num2str(top10_time_1_length(2)./ Mylength(2)));
    fprintf(fid,'distance20 = %s\n' ,num2str(top10_time_1_length(3)./ Mylength(3)));
    fprintf(fid,'distance50 = %s\n' ,num2str(top10_time_1_length(4)./ Mylength(4)));
    fprintf(fid,'distance100 =%s\n' ,num2str(top10_time_1_length(5)./ Mylength(5)));
    fprintf(fid,'distance200 = %s\n' ,num2str(top10_time_1_length(6)./ Mylength(6)));
    fprintf(fid,'distance500 = %s\n' ,num2str(top10_time_1_length(7)./ Mylength(7)));
    fprintf(fid,'distance800 = %s\n' ,num2str(top10_time_1_length(8)./ Mylength(8)));
    fprintf(fid,'\n');
    
    top10_time_1_length_quantitative=top10_time_1_length_cum ./ Mylength_cum;
    fprintf(fid,'distance_quantitative5 = %s\n' ,num2str(top10_time_1_length_quantitative(1)));
    fprintf(fid,'distance_quantitative10 = %s\n' ,num2str(top10_time_1_length_quantitative(2)));
    fprintf(fid,'distance_quantitative20 = %s\n' ,num2str(top10_time_1_length_quantitative(3)));
    fprintf(fid,'distance_quantitative50 = %s\n' ,num2str(top10_time_1_length_quantitative(4)));
    fprintf(fid,'distance_quantitative100 =%s\n' ,num2str(top10_time_1_length_quantitative(5)));
    fprintf(fid,'distance_quantitative200 = %s\n' ,num2str(top10_time_1_length_quantitative(6)));
    fprintf(fid,'distance_quantitative500 = %s\n' ,num2str(top10_time_1_length_quantitative(7)));
    fprintf(fid,'distance_quantitative800 = %s\n' ,num2str(top10_time_1_length_quantitative(8)));
    fprintf(fid,'\n');
    %end(top10_time_1_distance_quantitative)
    
    %time_quantitative
    right_time(1)= length(find(right_when>=0&right_when< (900/time_metric) ));
    right_time(2)= length(find(right_when>= (900/time_metric) &right_when< (1800/time_metric) ));
    right_time(3)= length(find(right_when>= (1800/time_metric) &right_when< (3600/time_metric) ));
    right_time(4)= length(find(right_when>= (3600/time_metric) &right_when< (21600/time_metric) ));
    right_time(5)= length(find(right_when>= (21600/time_metric) &right_when< (43200/time_metric) ));
    right_time(6)= length(find(right_when>= (43200/time_metric) &right_when< (86400/time_metric) ));
    right_time(7)= length(find(right_when>= (86400/time_metric) &right_when< (172800/time_metric) ));
    right_time(8)= length(find(right_when>= (172800/time_metric) &right_when< (604800/time_metric) ));
    right_time_cum=cumsum(right_time);
    
    fprintf(fid,'quarter = %s\n' ,num2str(right_time(1)./ Mytime(1)));
    fprintf(fid,'half = %s\n' ,num2str(right_time(2)./ Mytime(2)));
    fprintf(fid,'time1 = %s\n' ,num2str(right_time(3)./ Mytime(3)));
    fprintf(fid,'time6 = %s\n' ,num2str(right_time(4)./ Mytime(4)));
    fprintf(fid,'time12 = %s\n' ,num2str(right_time(5)./ Mytime(5)));
    fprintf(fid,'time24 = %s\n' ,num2str(right_time(6)./ Mytime(6)));
    fprintf(fid,'time48 = %s\n' ,num2str(right_time(7)./ Mytime(7)));
    fprintf(fid,'time168 = %s\n' ,num2str(right_time(8)./ Mytime(8)));
    fprintf(fid,'\n');
    
    time_quantitative=right_time_cum ./ Mytime_cum;
    fprintf(fid,'time_quantitative_quarter = %s\n' ,num2str(time_quantitative(1)));
    fprintf(fid,'time_quantitative_half = %s\n' ,num2str(time_quantitative(2)));
    fprintf(fid,'time_quantitative1 = %s\n' ,num2str(time_quantitative(3)));
    fprintf(fid,'time_quantitative6 = %s\n' ,num2str(time_quantitative(4)));
    fprintf(fid,'time_quantitative12 = %s\n' ,num2str(time_quantitative(5)));
    fprintf(fid,'time_quantitative24 = %s\n' ,num2str(time_quantitative(6)));
    fprintf(fid,'time_quantitative48 = %s\n' ,num2str(time_quantitative(7)));
    fprintf(fid,'time_quantitative168 = %s\n' ,num2str(time_quantitative(8)));
    fprintf(fid,'\n');
    %end(time_quantitative)
    
    %top10_time_1_when_quantitative
    top10_time_1_time(1)= length(find(top10_time_1_when>=0&top10_time_1_when< (900/time_metric) ));
    top10_time_1_time(2)= length(find(top10_time_1_when>= (900/time_metric) &top10_time_1_when< (1800/time_metric) ));
    top10_time_1_time(3)= length(find(top10_time_1_when>= (1800/time_metric) &top10_time_1_when< (3600/time_metric) ));
    top10_time_1_time(4)= length(find(top10_time_1_when>= (3600/time_metric) &top10_time_1_when< (21600/time_metric) ));
    top10_time_1_time(5)= length(find(top10_time_1_when>= (21600/time_metric) &top10_time_1_when< (43200/time_metric) ));
    top10_time_1_time(6)= length(find(top10_time_1_when>= (43200/time_metric) &top10_time_1_when< (86400/time_metric) ));
    top10_time_1_time(7)= length(find(top10_time_1_when>= (86400/time_metric) &top10_time_1_when< (172800/time_metric) ));
    top10_time_1_time(8)= length(find(top10_time_1_when>= (172800/time_metric) &top10_time_1_when< (604800/time_metric) ));
    top10_time_1_time_cum=cumsum(top10_time_1_time);
    
    fprintf(fid,'top10_time_1_quarter = %s\n' ,num2str(top10_time_1_time(1)./ Mytime(1)));
    fprintf(fid,'top10_time_1_half = %s\n' ,num2str(top10_time_1_time(2)./ Mytime(2)));
    fprintf(fid,'top10_time_1_time1 = %s\n' ,num2str(top10_time_1_time(3)./ Mytime(3)));
    fprintf(fid,'top10_time_1_time6 = %s\n' ,num2str(top10_time_1_time(4)./ Mytime(4)));
    fprintf(fid,'top10_time_1_time12 = %s\n' ,num2str(top10_time_1_time(5)./ Mytime(5)));
    fprintf(fid,'top10_time_1_time24 = %s\n' ,num2str(top10_time_1_time(6)./ Mytime(6)));
    fprintf(fid,'top10_time_1_time48 = %s\n' ,num2str(top10_time_1_time(7)./ Mytime(7)));
    fprintf(fid,'top10_time_1_time168 = %s\n' ,num2str(top10_time_1_time(8)./ Mytime(8)));
    fprintf(fid,'\n');
    
    top10_time_1_time_quantitative=top10_time_1_time_cum ./ Mytime_cum;
    fprintf(fid,'top10_time_1_time_quantitative_quarter = %s\n' ,num2str(top10_time_1_time_quantitative(1)));
    fprintf(fid,'top10_time_1_time_quantitative_half = %s\n' ,num2str(top10_time_1_time_quantitative(2)));
    fprintf(fid,'top10_time_1_time_quantitative1 = %s\n' ,num2str(top10_time_1_time_quantitative(3)));
    fprintf(fid,'top10_time_1_time_quantitative6 = %s\n' ,num2str(top10_time_1_time_quantitative(4)));
    fprintf(fid,'top10_time_1_time_quantitative12 = %s\n' ,num2str(top10_time_1_time_quantitative(5)));
    fprintf(fid,'top10_time_1_time_quantitative24 = %s\n' ,num2str(top10_time_1_time_quantitative(6)));
    fprintf(fid,'top10_time_1_time_quantitative48 = %s\n' ,num2str(top10_time_1_time_quantitative(7)));
    fprintf(fid,'top10_time_1_time_quantitative168 = %s\n' ,num2str(top10_time_1_time_quantitative(8)));
    fprintf(fid,'\n');
    %end(top10_time_1_when_quantitative)
    
     %top20_time_1_when_quantitative
    top20_time_1_time(1)= length(find(top20_time_1_when>=0&top20_time_1_when< (900/time_metric) ));
    top20_time_1_time(2)= length(find(top20_time_1_when>= (900/time_metric) &top20_time_1_when< (1800/time_metric) ));
    top20_time_1_time(3)= length(find(top20_time_1_when>= (1800/time_metric) &top20_time_1_when< (3600/time_metric) ));
    top20_time_1_time(4)= length(find(top20_time_1_when>= (3600/time_metric) &top20_time_1_when< (21600/time_metric) ));
    top20_time_1_time(5)= length(find(top20_time_1_when>= (21600/time_metric) &top20_time_1_when< (43200/time_metric) ));
    top20_time_1_time(6)= length(find(top20_time_1_when>= (43200/time_metric) &top20_time_1_when< (86400/time_metric) ));
    top20_time_1_time(7)= length(find(top20_time_1_when>= (86400/time_metric) &top20_time_1_when< (172800/time_metric) ));
    top20_time_1_time(8)= length(find(top20_time_1_when>= (172800/time_metric) &top20_time_1_when< (604800/time_metric) ));
    top20_time_1_time_cum=cumsum(top20_time_1_time);
    
    fprintf(fid,'top20_time_1_quarter = %s\n' ,num2str(top20_time_1_time(1)./ Mytime(1)));
    fprintf(fid,'top20_time_1_half = %s\n' ,num2str(top20_time_1_time(2)./ Mytime(2)));
    fprintf(fid,'top20_time_1_time1 = %s\n' ,num2str(top20_time_1_time(3)./ Mytime(3)));
    fprintf(fid,'top20_time_1_time6 = %s\n' ,num2str(top20_time_1_time(4)./ Mytime(4)));
    fprintf(fid,'top20_time_1_time12 = %s\n' ,num2str(top20_time_1_time(5)./ Mytime(5)));
    fprintf(fid,'top20_time_1_time24 = %s\n' ,num2str(top20_time_1_time(6)./ Mytime(6)));
    fprintf(fid,'top20_time_1_time48 = %s\n' ,num2str(top20_time_1_time(7)./ Mytime(7)));
    fprintf(fid,'top20_time_1_time168 = %s\n' ,num2str(top20_time_1_time(8)./ Mytime(8)));
    fprintf(fid,'\n');
    
    top20_time_1_time_quantitative=top20_time_1_time_cum ./ Mytime_cum;
    fprintf(fid,'top20_time_1_time_quantitative_quarter = %s\n' ,num2str(top20_time_1_time_quantitative(1)));
    fprintf(fid,'top20_time_1_time_quantitative_half = %s\n' ,num2str(top20_time_1_time_quantitative(2)));
    fprintf(fid,'top20_time_1_time_quantitative1 = %s\n' ,num2str(top20_time_1_time_quantitative(3)));
    fprintf(fid,'top20_time_1_time_quantitative6 = %s\n' ,num2str(top20_time_1_time_quantitative(4)));
    fprintf(fid,'top20_time_1_time_quantitative12 = %s\n' ,num2str(top20_time_1_time_quantitative(5)));
    fprintf(fid,'top20_time_1_time_quantitative24 = %s\n' ,num2str(top20_time_1_time_quantitative(6)));
    fprintf(fid,'top20_time_1_time_quantitative48 = %s\n' ,num2str(top20_time_1_time_quantitative(7)));
    fprintf(fid,'top20_time_1_time_quantitative168 = %s\n' ,num2str(top20_time_1_time_quantitative(8)));
    fprintf(fid,'iteration = %s\n',num2str(iteration_i));
    fprintf(fid,'\n');
    %end(top20_time_1_when_quantitative)
    
    %next new POI
    right_case_1=right_i_1>-1;
    right_case_1=right_case_1';
    right_case_1=testset(right_case_1,:);
    Nextnew_1=arrayfun(@(i) any(trainset(:,1)==right_case_1(i,1))&(any(trainset(:,2)==right_case_1(i,3))|any(trainset(:,3)==right_case_1(i,3))),1:size(right_case_1,1));

    right_case_5=right_i_5>-1;
    right_case_5=right_case_5';
    right_case_5=testset(right_case_5,:);
    Nextnew_5=arrayfun(@(i) any(trainset(:,1)==right_case_5(i,1))&(any(trainset(:,2)==right_case_5(i,3))|any(trainset(:,3)==right_case_5(i,3))),1:size(right_case_5,1));

    right_case_10=right_i_10>-1;
    right_case_10=right_case_10';
    right_case_10=testset(right_case_10,:);
    Nextnew_10=arrayfun(@(i) any(trainset(:,1)==right_case_10(i,1))&(any(trainset(:,2)==right_case_10(i,3))|any(trainset(:,3)==right_case_10(i,3))),1:size(right_case_10,1));

    right_case_20=right_i_20>-1;
    right_case_20=right_case_20';
    right_case_20=testset(right_case_20,:);
    Nextnew_20=arrayfun(@(i) any(trainset(:,1)==right_case_20(i,1))&(any(trainset(:,2)==right_case_20(i,3))|any(trainset(:,3)==right_case_20(i,3))),1:size(right_case_20,1));
    
    right_case_50=right_i_50>-1;
    right_case_50=right_case_50';
    right_case_50=testset(right_case_50,:);
    Nextnew_50=arrayfun(@(i) any(trainset(:,1)==right_case_50(i,1))&(any(trainset(:,2)==right_case_50(i,3))|any(trainset(:,3)==right_case_50(i,3))),1:size(right_case_50,1));
    
    fprintf(fid,'NextNewPOI_1=%s\n',num2str((size(Nextnew_1,2) - sum(Nextnew_1))./ row_testset));
    fprintf(fid,'NextNewPOI_5=%s\n',num2str((size(Nextnew_5,2) - sum(Nextnew_5))./ row_testset));
    fprintf(fid,'NextNewPOI_10=%s\n',num2str((size(Nextnew_10,2) - sum(Nextnew_10))./ row_testset));
    fprintf(fid,'NextNewPOI_20=%s\n',num2str((size(Nextnew_20,2) - sum(Nextnew_20))./ row_testset));
    fprintf(fid,'NextNewPOI_50=%s\n',num2str((size(Nextnew_50,2) - sum(Nextnew_50))./ row_testset));
    
    fprintf(fid,'iteration = %s\n',num2str(iteration_i));
    fprintf(fid,'\n');
    %end(next new POI)
    
end %while 1

fclose(fid);


