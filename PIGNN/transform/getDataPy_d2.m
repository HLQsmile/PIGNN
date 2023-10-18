%% Load lib file
if ispc 
    addpath('...\PIGNN')
end

if isunix
    addpath('..\PIGNN')
end

mkdir('...\PIGNN\train\');
%% set parameter
Parm.Method = ['tSNE'];              % Alternative methods: 'kpca' or 'pca' 
Parm.Max_Px_Size =400;               % Size of image
Parm.MPS_Fix = 1;
Parm.ValidRatio = 0.1;               % ratio of validation data/Training data
Parm.Seed = 108;                     % random seed to distribute training and validation sets
Parm.K = 11;                         % number of sampling
Parm.NORM = 2;
%%
load('...\PIGNN\train.mat');
dataset=train;
dataset.Xtrain=dataset.Xtrain;


%% Get feature matrix and image datasets 
all( dataset.Xtrain == 0,2);
dataset.Xtrain(all(dataset.Xtrain  == 0,2),:) = [];
Out=getMatrix(dataset,Parm);

Tip = 'getMatrix done ...';

%%
fid = fopen('..\PIGNN\train_label.txt','a')
timeFolder='..\PIGNN\train\'

Labs = unique(string(dataset.train_labels));
Pa={};
for i=1:length(Labs)
     mkdir([timeFolder,'\'],Labs{i});
     P=strcat(timeFolder,Labs{i},'\')
     Pa{i}=P
end
%%
data_size = size(Out.Xtrain);
for i = 1:1:data_size(4)
    switch char(Out.train_labels(i))
        case Labs(1)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{1},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);

            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(2)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{2},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(3)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{3},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(4)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{4},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(5)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{5},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(6)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{6},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);

            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(7)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{7},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(8)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{8},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n'); 
        case Labs(9)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{9},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(10)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{10},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
        case Labs(11)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{11},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);
    
            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
         case Labs(12)
            Data = Out.Xtrain(:,:,:,i);
            Data = repmat(Data,[1 1 3]);
            image_filename = strcat(Pa{12},sprintf('%d',i),'.jpg');
            imwrite(Data, image_filename);

            filename = strcat(sprintf('%d',i),'.jpg');
            fprintf(fid,'%s',filename); 
            fprintf(fid,'\t%s',string(find(Labs==string(Out.train_labels(i)))));
            fprintf(fid,'\r\n');
    end
end
fclose(fid); 
