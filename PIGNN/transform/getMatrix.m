function Out = getMatrix(dset,Parm)

    if exist('Parm')==0             % Default parameter setting
       Parm.Method = ['tSNE'];      % Other methods: 'kpca' or 'pca' 
       Parm.Max_Px_Size = 300;      % Size of image
       Parm.MPS_Fix = 1;            % if this val is 1 then screen will be Max_Px_Size x Max_Px_Size (eg 120x120 )
                                    % otherwise automatically decided by the distribution
                                    % of the input data.
       Parm.ValidRatio = 0.1;       % 0.1 of Train data will be used as Validation data
       Parm.Seed = 108;             % Random seed
    end 

    YTrain=categorical(dset.train_labels)';
    TrueLabel = YTrain;
    class = unique(YTrain);
    q=1:length(TrueLabel);
    clear idx
    rng('default')
    idx={};
    for j=1:length(class)
        index = ismember(dset.train_labels,class(j));
        ri=find(index==1);
        rng('shuffle')
%         idx{j} = ri(randperm(length(ri),round(length(ri)*Parm.ValidRatio)));
    end
    idx=cell2mat(idx);
%     dset.XValidation = dset.Xtrain(:,idx);
    dset.Xtrain(:,idx) = [];
%     YValidation = YTrain(idx);
    YTrain(idx) = [];

    %%
    switch Parm.NORM
        case 1
            % Norm-3 in org code
            Out.Norm=1;
            fprintf('\nNORM-1\n');
            %########### Norm-1 ###################
            Out.Max=max(dset.Xtrain')';
            Out.Min=min(dset.Xtrain')';
            dset.Xtrain=(dset.Xtrain-Out.Min)./(Out.Max-Out.Min);
%             dset.XValidation = (dset.XValidation-Out.Min)./(Out.Max-Out.Min);
            dset.Xtrain(isnan(dset.Xtrain))=0;
%             dset.XValidation(isnan(dset.XValidation))=0;
%             dset.XValidation(dset.XValidation>1)=1;
%             dset.XValidation(dset.XValidation<0)=0;
            %######################################

        case 2
            % norm-6 in org ocde
            Out.Norm=2;

            fprintf('\nNORM-2\n');
            %########### Norm-2 ###################
            Out.Min=min(dset.Xtrain')';
            dset.Xtrain=log(dset.Xtrain+abs(Out.Min)+1);
%             indV = dset.XValidation<Out.Min;
 
%             for j=1:size(dset.Xtrain,1)
%                 dset.XValidation(j,indV(j,:))=Out.Min(j);
%             end
%             dset.XValidation = log(dset.XValidation+abs(Out.Min)+1);
            Out.Max=max(max(dset.Xtrain));
            dset.Xtrain=dset.Xtrain/Out.Max;
%             dset.XValidation = dset.XValidation/Out.Max;
%             dset.XValidation(dset.XValidation>1)=1;
            %######################################
    end

    Q.data = dset.Xtrain;
    Q.Method = Parm.Method;%['tSNE'];
    Q.Max_Px_Size = Parm.Max_Px_Size;%120;
%     Out.ValidationRawdata = dset.XValidation;

    if Parm.MPS_Fix==1
        [Out.M,Out.xp,Out.yp,Out.A,Out.B,Out.Base] = Cart2Pixel(Q,Q.Max_Px_Size,Q.Max_Px_Size);
    else
        [Out.M,Out.xp,Out.yp,Out.A,Out.B,Out.Base] = Cart2Pixel(Q);
    end
    
    fprintf('\n Pixels: %d x %d\n',Out.A,Out.B);
    clear Q
    dset.Xtrain=[];
    close all;

%     for j=1:length(YValidation)
%         XValidation(:,:,1,j) = ConvPixel(dset.XValidation(:,j),Out.xp,Out.yp,Out.A,Out.B,Out.Base,0);
%     end
    
%     dset.XValidation=[];
    for j=1:length(YTrain)
        XTrain(:,:,1,j) = Out.M{j};
    end

    clear M X Y
    
    Out.Xtrain = XTrain;
%     Out.XValidation =  XValidation;
    Out.train_labels = YTrain;
%     Out.Validation_labels= YValidation;
end