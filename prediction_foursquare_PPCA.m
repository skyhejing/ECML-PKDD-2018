function [Myresult_1,Myresult_5,Myresult_10,Myresult_20,Myresult_30,Myresult_40,Myresult_50,right_distance,right_when,Myresult_time_quarter,Myresult_time_half,Myresult_time_1,Myresult_time_6,Myresult_time_12,Myresult_time_24,Myresult_time_48,Myresult_time_168,Myresult_time_location_1,Myresult_time_location_5,Myresult_time_location_10,Myresult_time_location_20,Myresult_time_location_50,Myresult_location10_time_1,Myresult_location10_time_6,Myresult_location10_time_12,Myresult_location10_time_24,Myresult_location10_time_48,Myresult_location10_time_168,right_i_1,right_i_5,right_i_10,right_i_20,right_i_50,time_distance_MAE,time_distance_MAPE,top10_time_1_distance,top10_time_1_when,top20_time_1_when] = prediction_foursquare_PPCA( testset,x_l_u,x_u_l,x_l_p,x_p_l,z_l_u,z_u_l,z_l_p,z_p_l,rho,w,location_unique_test,distance_matrix_frac,time_metric,distance_metric_meter,i)
%UNTITLED2 Summary of this function goes here
    %init the varibale in if{if{}}
    Myresult_location10_time_1=0;
    Myresult_location10_time_6=0;
    Myresult_location10_time_12=0;
    Myresult_location10_time_24=0;
    Myresult_location10_time_48=0;
    Myresult_location10_time_168=0;
    
    Myresult_time_location_1=0;
    Myresult_time_location_5=0;
    Myresult_time_location_10=0;
    Myresult_time_location_20=0;
    Myresult_time_location_50=0;
    
    top10_time_1_distance=-1;
    top10_time_1_when=-1;
    top20_time_1_when=-1;
    %end(init the varibale in if{if{}})
    
    %deal with the x_tensor
    x_u_l_predict=x_u_l(testset(i,1),:);
    x_p_l_predict=x_p_l(testset(i,2),:);
    x_u_l_test=repmat(x_u_l_predict,[size(location_unique_test,1) 1]);
    x_p_l_test=repmat(x_p_l_predict,[size(location_unique_test,1) 1]);
    
    x_l_u_test=x_l_u(location_unique_test(:,1),:);
    x_l_p_test=x_l_p(location_unique_test(:,1),:);
    
    x_u_l_sum=sum(x_u_l_test .* x_l_u_test,2);
    x_p_l_sum=sum(x_p_l_test .* x_l_p_test,2);
    
    %for test
%     rho=10000;
    %end(for test)
    x_rho=rho;
    prelocation_line=location_unique_test(location_unique_test(:,1)==testset(i,2),4);
    distance_test_frac=distance_matrix_frac(prelocation_line,:);
    distance_test_frac = distance_test_frac ./ distance_metric_meter;
    distance_rho=x_rho .* distance_test_frac;
    
    x_tenosr=x_u_l_sum + x_p_l_sum + distance_rho';
    %end(deal with the x_tensor)
    
    %deal with the z_tensor
    z_u_l_predict=z_u_l(testset(i,1),:);
    z_p_l_predict=z_p_l(testset(i,2),:);
    z_u_l_test=repmat(z_u_l_predict,[size(location_unique_test,1) 1]);
    z_p_l_test=repmat(z_p_l_predict,[size(location_unique_test,1) 1]);
    
    z_l_u_test=z_l_u(location_unique_test(:,1),:);
    z_l_p_test=z_l_p(location_unique_test(:,1),:);
    
    z_u_l_sum=sum(z_u_l_test .* z_l_u_test,2);
    z_p_l_sum=sum(z_p_l_test .* z_l_p_test,2);
    
    z_tenosr=z_u_l_sum + z_p_l_sum;
    %end(eal with the z_tensor)
    
    %deal with the location's time
    z_u_l_time=z_u_l(testset(i,1),:);
    z_p_l_time=z_p_l(testset(i,2),:);
    z_l_u_time=z_l_u(testset(i,3),:);
    z_l_p_time=z_l_p(testset(i,3),:);
    
    z_u_l_time_sum=sum(z_u_l_time .* z_l_u_time,2);
    z_p_l_time_sum=sum(z_p_l_time .* z_l_p_time,2);
    
    z_time_tensor=z_u_l_time_sum + z_p_l_time_sum;
    z_time=1 / z_time_tensor;
    %end(deal with the location's time)
    
    %predict location performance
    x_w=w(testset(i,1),:);
    location_result= x_w .* z_tenosr + x_tenosr;
    
    [~, index_sum]=sort(location_result,'descend');
    index_sum_top1=index_sum(1);
    index_sum_top5=index_sum(1:5);
    index_sum_top10=index_sum(1:10);
    index_sum_top20=index_sum(1:20);
    index_sum_top30=index_sum(1:30);
    index_sum_top40=index_sum(1:40);
    index_sum_top50=index_sum(1:50);
    
    if any(testset(i,3)==location_unique_test(index_sum_top1,1))
        Myresult_1=1;
        right_i_1=i;
    else
        Myresult_1=0;
        right_i_1=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top5,1))
        Myresult_5=1;
        right_i_5=i;
    else
        Myresult_5=0;
        right_i_5=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top10,1))
        Myresult_10=1;
        right_i_10=i;
        right_distance=testset(i,8);
        right_when=testset(i,22);
        
        if abs(testset(i,22)-z_time)< (3600/time_metric)
            Myresult_location10_time_1=1;
            
            %20170703.quantitative for XinLi
            top10_time_1_distance=testset(i,8);
            top10_time_1_when=testset(i,22);
        end
        
        if abs(testset(i,22)-z_time)< (21600/time_metric)
            Myresult_location10_time_6=1;
        end
        
        if abs(testset(i,22)-z_time)< (43200/time_metric)
            Myresult_location10_time_12=1;
        end
        
        if abs(testset(i,22)-z_time)< (86400/time_metric)
            Myresult_location10_time_24=1;
        end
        
        if abs(testset(i,22)-z_time)< (172800/time_metric)
            Myresult_location10_time_48=1;
        end
        
        if abs(testset(i,22)-z_time)< (604800/time_metric)
            Myresult_location10_time_168=1;
        end
        
    else
        Myresult_10=0;
        right_i_10=-1;
        right_distance=-1;
        right_when=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top20,1))
        Myresult_20=1;
        right_i_20=i;
        
        if abs(testset(i,22)-z_time)< (3600/time_metric)
            top20_time_1_when=testset(i,22);
        end
    else
        Myresult_20=0;
        right_i_20=-1;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top30,1))
        Myresult_30=1;
    else
        Myresult_30=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top40,1))
        Myresult_40=1;
    else
        Myresult_40=0;
    end
    
    if any(testset(i,3)==location_unique_test(index_sum_top50,1))
        Myresult_50=1;
        right_i_50=i;
    else
        Myresult_50=0;
        right_i_50=-1;
    end
     %end(predict location performance)
     
    %predict time performance
    time_distance_MAE=abs(testset(i,22)-z_time);
    time_distance_MAPE=abs(testset(i,22)-z_time) ./ (testset(i,22)+1);
    
    if abs(testset(i,22)-z_time)< (900/time_metric)
        Myresult_time_quarter=1;
    else
        Myresult_time_quarter=0;
    end
    
    if abs(testset(i,22)-z_time)< (1800/time_metric)
        Myresult_time_half=1;
    else
        Myresult_time_half=0;
    end
    
    if abs(testset(i,22)-z_time)< (3600/time_metric)
        Myresult_time_1=1;
    else
        Myresult_time_1=0;
    end
    
    if abs(testset(i,22)-z_time)< (21600/time_metric)
        Myresult_time_6=1;
    else
        Myresult_time_6=0;
    end
    
    if abs(testset(i,22)-z_time)< (43200/time_metric)
        Myresult_time_12=1;
    else
        Myresult_time_12=0;
    end
    
    if abs(testset(i,22)-z_time)< (86400/time_metric)
        Myresult_time_24=1;
        
        if any(testset(i,3)==location_unique_test(index_sum_top1,1))
            Myresult_time_location_1=1;
        end
        
        if any(testset(i,3)==location_unique_test(index_sum_top5,1))
            Myresult_time_location_5=1;
        end
        
        if any(testset(i,3)==location_unique_test(index_sum_top10,1))
            Myresult_time_location_10=1;
        end
        
        if any(testset(i,3)==location_unique_test(index_sum_top20,1))
            Myresult_time_location_20=1;
        end
        
        if any(testset(i,3)==location_unique_test(index_sum_top50,1))
            Myresult_time_location_50=1;
        end
    
    else
        Myresult_time_24=0;
    end
    
    if abs(testset(i,22)-z_time)< (172800/time_metric)
        Myresult_time_48=1;
    else
        Myresult_time_48=0;
    end
    
    if abs(testset(i,22)-z_time)< (604800/time_metric)
        Myresult_time_168=1;
    else
        Myresult_time_168=0;
    end
     %end(predict time performance)
end