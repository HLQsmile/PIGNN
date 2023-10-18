function allData_smote = getSMOTE(data,label,k)
    % getSMOTE  Synthetic Minority Oversampling Technique. A technique to
    % generate synthetic samples as given in: https://www.jair.org/media/953/live-953-2037-jair.pdf
    %   Usage:
    %   X_smote = mySMOTE(X, N, k) 
    %   
    %   Inputs:
    %   allData: Original dataset
    %   k: number of nearest neighbors to consider while performing
    %   augmentation
    %   sortedIDX: sorted labels
    % 
    %   Outputs:
    %   X_smote: augmented dataset containing original data as well.
    %   
    %   See also datasample, randsample
    %% plot the bar plot for number of classes

    %% number of each classes
    labels=label;
    class=unique(labels);
    for ii=1:numel(class)
        classNo(ii)=numel(find(ismember(labels,class(ii))==1));
    end
    
    %% required addon samples in each minority class
    %add on samples will be calculated by taking the difference of each
    %classSamples with highest number of class sample
    [maximumSamples,sampleClass]=max(classNo); % number of maximum samples
    for ii=1:numel(class)
        X=data(find(ismember(labels,class(ii))==1),:);
        T = size(X, 1);
        samplediff(ii)=maximumSamples-classNo(ii);
        N (ii) = round(samplediff(ii)/ T);
    end
    
    %% oversample the minority classes
    Xdata =[];
    Xlabels ={};
    for ii=1:numel(class)
        X=data(find(ismember(labels,class(ii))==1),:);
        T = size(X, 1);
        X_smote = X;
        X_labels = labels(find(ismember(labels,class(ii))==1),1);
        if N(ii) > 0
            for i = 1:T
                x_labels = {}; 
                y = X(i,:);
                % find k-nearest samples
                [idx, ~] = knnsearch(X,y,'k',k);
                % retain only N out of k nearest samples
                idx = datasample(idx, N(ii));
                x_nearest = X(idx,:);
                x_syn = bsxfun(@plus, bsxfun(@times, bsxfun(@minus,x_nearest,y), rand(N(ii),1)), y);
                X_smote = cat(1, X_smote,x_syn);
                [s,~] = size(x_syn);
                x_labels(1:s,1) = class(ii); 
                X_labels = cat(1, X_labels, x_labels);
            end
            ii
        end
        Xdata = cat(1,Xdata,X_smote);
        Xlabels = cat(1,Xlabels ,X_labels);
    end
    
    allData_smote.Xtrain = Xdata';
    allData_smote.train_labels = Xlabels';
end

