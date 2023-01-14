%% Load lib file
if ispc 
    addpath('')
end

if isunix
    addpath('')
end

mkdir('');


%% set parameter
Parm.Method = ['tSNE'];              % Alternative methods: 'kpca' or 'pca' 
Parm.Max_Px_Size =400;              % Size of image
Parm.MPS_Fix = 1;
Parm.ValidRatio = 0.1;               % ratio of validation data/Training data
Parm.Seed = 108;                     % random seed to distribute training and validation sets
Parm.K = 11;                          % number of sampling
Parm.NORM = 2;
%% 
load('\train.mat');
D=train;
D.Xtrain=D.Xtrain;
% Get balanced data
dset =  getSMOTE(D.Xtrain',D.train_labels',Parm.K); 
Labs = unique(string(D.train_labels));
Tip = 'SMOTE done ...';

%% Get feature matrix and image datasets 
all( dset.Xtrain == 0,2);
dset.Xtrain(all(dset.Xtrain  == 0,2),:) = [];
Out = getMatrix(dset,Parm);
Tip = 'getMatrix done ...';
%%
fid = fopen('\train_label.txt','a')
% ss = size(Out.Xtrain);
save_path='C:\Users\DELL\Desktop\PIGNN\data\train\class1\';
% rmdir(save_path,'s');
mkdir(save_path);
save_path='C:\Users\DELL\Desktop\PIGNN\data\train\Class2\';
% rmdir(save_path,'s');
mkdir(save_path);
save_path='C:\Users\DELL\Desktop\PIGNN\data\train\Class3\';
% rmdir(save_path,'s');
mkdir(save_path);
save_path='C:\Users\DELL\Desktop\PIGNN\data\train\Class4\';
% rmdir(save_path,'s');
mkdir(save_path);
ss = size(Out.Xtrain);
for i = 1:1:ss(4)
    switch char(Out.train_labels(i))
        case Labs(1)
            DD = Out.Xtrain(:,:,:,i);
            DD = repmat(DD,[1 1 3]);
            sc = strcat('C:\Users\DELL\Desktop\PIGNN\data\train\Class1\',sprintf('%d',i));
            sc = strcat(sc,'.jpg');
            imwrite(DD, sc);

            fn = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',fn); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(2)
            DD = Out.Xtrain(:,:,:,i);
            DD = repmat(DD,[1 1 3]);
            sc = strcat('C:\Users\DELL\Desktop\PIGNN\data\train\Class2\',sprintf('%d',i));
            sc = strcat(sc,'.jpg');
            imwrite(DD, sc);
    
            fn = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',fn); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(3)
            DD = Out.Xtrain(:,:,:,i);
            DD = repmat(DD,[1 1 3]);
            sc = strcat('C:\Users\DELL\Desktop\PIGNN\data\train\Class3\',sprintf('%d',i));
            sc = strcat(sc,'.jpg');
            imwrite(DD, sc);
    
            fn = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',fn); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(4)
            DD = Out.Xtrain(:,:,:,i);
            DD = repmat(DD,[1 1 3]);
            sc = strcat('C:\Users\DELL\Desktop\PIGNN\data\train\Class4\',sprintf('%d',i));
            sc = strcat(sc,'.jpg');
            imwrite(DD, sc);
    
            fn = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',fn); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
    end
    fprintf('%d images\n',i)
end
fprintf("finished\n")
fclose(fid);

