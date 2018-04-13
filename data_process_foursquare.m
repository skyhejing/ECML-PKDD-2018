%get a new dataset. This dataset only contains the distance information.

dataCase=dlmread('userTip_foursquare.txt');
% dataCase=dlmread('userTip_foursquare.txt');
%To delete the longitude and latitude.
% a=dataCase(:,1:2);
% b=dataCase(:,5:10);
% c=[a b];
%Delete this line to improve the precision.
%c_unique=unique(c,'rows','stable');
user_unique=unique(dataCase(:,1));
loc_lon_lat=dataCase(:,2:4);
location_unique=unique(loc_lon_lat,'rows');

% user_line=1:2823;
% user_line=user_line';
% user_unique_line=[user_unique user_line];
% 
% location_line=1:91787;
% location_line=location_line';
% location_unique_line=[location_unique location_line];

row_dataCase=size(dataCase,1);

Mydata=zeros(row_dataCase-1,11);

%Because the first line of the dataset don't have the previous location
for i=1:row_dataCase-1
    if dataCase(i,1)==dataCase(i+1,1)
        %userid
        Mydata(i,1)=dataCase(i+1,1);
        %previous location
        Mydata(i,2)=dataCase(i,2);
        %current location or next location
        Mydata(i,3)=dataCase(i+1,2);
        %previous location's longitude
        Mydata(i,4)=dataCase(i,3);
        %previous location's latitude
        Mydata(i,5)=dataCase(i,4);
        %next location's longitude
        Mydata(i,6)=dataCase(i+1,3);
        %next location's latitude
        Mydata(i,7)=dataCase(i+1,4);
        if 6==dataCase(i+1,10) || 7==dataCase(i+1,10)
            %weekend
            Mydata(i,9)=-1;
        else
            %weekday
            Mydata(i,9)=1;
        end
        
        if dataCase(i+1,8)>7 && dataCase(i+1,8)<18
            %work time
            Mydata(i,10)=1;
        else
            %home time
            Mydata(i,10)=-1;
        end
        %category
        Mydata(i,11)=dataCase(i,11);
        
        %20151127. Camera Ready
        Mydata(i,12)=dataCase(i,5);
        Mydata(i,13)=dataCase(i,6);
        Mydata(i,14)=dataCase(i,7);
        Mydata(i,15)=dataCase(i,8);
        Mydata(i,16)=dataCase(i,9);
        
        Mydata(i,17)=dataCase(i+1,5);
        Mydata(i,18)=dataCase(i+1,6);
        Mydata(i,19)=dataCase(i+1,7);
        Mydata(i,20)=dataCase(i+1,8);
        Mydata(i,21)=dataCase(i+1,9);
        
    end
end 

%replace user id with the line number. so do location id
% tic
% [row_Mydata, col_Mydata]=size(Mydata);
% Mydata_line=zeros(row_Mydata,col_Mydata);
% for j=1:row_Mydata
%     if Mydata(j,1)~=0
%         Mydata_line(j,1)=find(user_unique==Mydata(j,1),1);
%         Mydata_line(j,2)=find(location_unique==Mydata(j,2),1);
%         Mydata_line(j,3)=find(location_unique==Mydata(j,3),1);
%         Mydata_line(j,4)=Mydata(j,4);
%         Mydata_line(j,5)=Mydata(j,5);
%     end
% end
% toc

%user map to replace user id with the line number. so do location id. This
%function can be solved by B = A(row_index,:).
%This function can also be solved by user_unique=unique(c_unique(:,1),'stable');
tic
[row_Mydata, col_Mydata]=size(Mydata);
Mydata_line=zeros(row_Mydata,col_Mydata);
userMap = containers.Map(user_unique(:,1),1:size(user_unique,1));
locationMap=containers.Map(location_unique(:,1),1:size(location_unique,1));
for j=1:row_Mydata
    if Mydata(j,1)~=0
        Mydata_line(j,1)=userMap(Mydata(j,1));
        Mydata_line(j,2)=locationMap(Mydata(j,2));
        Mydata_line(j,3)=locationMap(Mydata(j,3));
        Mydata_line(j,4)=Mydata(j,4);
        Mydata_line(j,5)=Mydata(j,5);
        Mydata_line(j,6)=Mydata(j,6);
        Mydata_line(j,7)=Mydata(j,7);
        Mydata_line(j,8)=Mydata(j,8);
        Mydata_line(j,9)=Mydata(j,9);
        Mydata_line(j,10)=Mydata(j,10);
        Mydata_line(j,11)=Mydata(j,11);
        %20151127. Camera Ready
        Mydata_line(j,12)=Mydata(j,12);
        Mydata_line(j,13)=Mydata(j,13);
        Mydata_line(j,14)=Mydata(j,14);
        Mydata_line(j,15)=Mydata(j,15);
        Mydata_line(j,16)=Mydata(j,16);
        Mydata_line(j,17)=Mydata(j,17);
        Mydata_line(j,18)=Mydata(j,18);
        Mydata_line(j,19)=Mydata(j,19);
        Mydata_line(j,20)=Mydata(j,20);
        Mydata_line(j,21)=Mydata(j,21);
    end
end
toc

Mydistance=arrayfun(@(i) distance(Mydata_line(i,4),Mydata_line(i,5),Mydata_line(i,6),Mydata_line(i,7),6378.1),1:size(Mydata_line,1));
Mydata_line(:,8)=Mydistance';

Mydtime=arrayfun(@(i) etime([Mydata_line(i,17) Mydata_line(i,18) Mydata_line(i,19) Mydata_line(i,20) Mydata_line(i,21) 00],[Mydata_line(i,12) Mydata_line(i,13) Mydata_line(i,14) Mydata_line(i,15) Mydata_line(i,16) 00]),1:size(Mydata_line,1));
Mydata_line(:,22)=Mydtime';



%cut the dataset into trainset and testset
tic
[row_Mydata_line, col_Mydata_line]=size(Mydata_line);
trainset_orginal=zeros(row_Mydata_line, col_Mydata_line);
testset=zeros(row_Mydata_line, col_Mydata_line);
zero_matrix=zeros(1,col_Mydata_line);
%zero is the sign that indicates a new user
Mydata_line=[Mydata_line;zero_matrix];
temp=0;
for k=1:row_Mydata+1
    if Mydata_line(k,1)~=0
        temp=temp+1;
    else
        cut=round(temp*0.8);
        trainset_orginal(k-temp:k-temp+cut-1,:)=Mydata_line(k-temp:k-temp+cut-1,:);
        testset(k-temp+cut:k-1,:)=Mydata_line(k-temp+cut:k-1,:);
        temp=0;
    end
end
toc

%delete the zero line.
trainset_orginal=trainset_orginal(any(trainset_orginal'),:);
trainset_sort=sortrows(trainset_orginal);

%calcute the average time and transion times
[row_trainset_sort, col_trainset_sort]=size(trainset_sort);
trainset=zeros(row_trainset_sort,col_trainset_sort+1);
%initialize the transion times
% trainset(:,col_trainset_sort+1)=ones(row_trainset_sort,1);
i=1;
while i<row_trainset_sort
    if (trainset_sort(i,1)==trainset_sort(i+1,1)) && (trainset_sort(i,2)==trainset_sort(i+1,2)) && (trainset_sort(i,3)==trainset_sort(i+1,3))
        trainset(i,1:22)=trainset_sort(i,:);
        trainset(i,23)=1;
        j=i+1;
        while (trainset_sort(i,1)==trainset_sort(j,1)) && (trainset_sort(i,2)==trainset_sort(j,2)) && (trainset_sort(i,3)==trainset_sort(j,3))
            %deal with the time
            trainset(i,22)=trainset(i,22) + trainset_sort(j,22);
            %deal with the transion tiems
            trainset(i,23)=trainset(i,23) +1;
            j=j+1;
        end
        i=j;
    else
        trainset(i,1:22)=trainset_sort(i,:);
        trainset(i,23)=1;
        i=i+1;
    end
end
%add the last line.
trainset(row_trainset_sort,1:22)=trainset_sort(row_trainset_sort,:);
trainset(row_trainset_sort,23)=1;

%the old version
% [unique_trainset_sort, iorginal, iunique]= unique(trainset_sort,'rows');
% count = hist(iunique,unique(iunique));
% unique_trainset_sort(:,23)=count';
% list_three=unique_trainset_sort(1<count,:);
% dlmwrite('J:\trainset_sort.txt', trainset_sort, 'precision', '%3.4f', 'delimiter', '\t');
%the old version. end

% trainset=trainset(:,1:11);
trainset=trainset(any(trainset'),:);
trainset(:,22)=trainset(:,22) ./ trainset(:,23);

[row_trainset, col_trainset]=size(trainset);
trainset_index=1:row_trainset;
trainset(:,24)=trainset_index';

testset=testset(any(testset'),:);

loc_lon_lat_testset_all=testset(:,2:5);
a=loc_lon_lat_testset_all(:,1);
b=loc_lon_lat_testset_all(:,3:4);
loc_lon_lat_testset=[a b];
location_unique_test=unique(loc_lon_lat_testset,'rows');
d=1:size(location_unique_test,1);
location_unique_test(:,4)=d';
location_unique_test_store=location_unique_test(:,2:3);
dlmwrite('D:\locs.txt', location_unique_test_store, 'precision', '%3.4f', 'delimiter', '\t');

tic
distance_matrix_frac=load('distance_matrix.mat');
distance_matrix_frac=struct2cell(distance_matrix_frac);
distance_matrix_frac=cell2mat(distance_matrix_frac);
toc

%  save data_foursquare;