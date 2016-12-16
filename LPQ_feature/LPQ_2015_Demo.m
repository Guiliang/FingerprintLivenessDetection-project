methodName = '2015_LPQ';
collectorName = 'Hi_Scan';
dataSetName = 'Hi_Scan';

%count the real data in training
train_real_dir = strcat('./ld-2015_data/Training/',dataSetName,'/Live');
train_real_num = length(dir([train_real_dir,'/*.bmp']));
if train_real_num == 0
    train_real_num = length(dir([train_real_dir,'/*.png']));
end
%count the real data in testing
test_real_dir = strcat('./ld-2015_data/Testing/',dataSetName,'/Live');
test_real_num = length(dir([test_real_dir,'/*.bmp']));
if test_real_num == 0
    test_real_num = length(dir([test_real_dir,'/*.png']));
end

%find the spoof method in training data
train_Spoof_dir = strcat('./ld-2015_data/Training/',dataSetName,'/Fake');
cell = struct2cell(dir([train_Spoof_dir,'/*']));
Spoof_Method_train = {};

Spoof_Method_train_count = 1;
for a = 1:length(cell)
    num = 1+5*(a-1);
    p = strfind(cell(num),'.');
    p_mat = cell2mat(p);
    if(isempty(p_mat))
        Spoof_Method_train{Spoof_Method_train_count} = cell2mat(cell(num));
        Spoof_Method_train_count = Spoof_Method_train_count+1;
    else
        continue
    end
end

%find the spoof method in training data
test_Spoof_dir = strcat('./ld-2015_data/testing/',dataSetName,'/Fake');
cell = struct2cell(dir([test_Spoof_dir,'/*']));
Spoof_Method_test = {};

Spoof_Method_test_count = 1;
for a = 1:length(cell)
    num = 1+5*(a-1);
    p = strfind(cell(num),'.');
    p_mat = cell2mat(p);
    if(isempty(p_mat))
        Spoof_Method_test{Spoof_Method_test_count} = cell2mat(cell(num));
        Spoof_Method_test_count = Spoof_Method_test_count+1;
    else
        continue
    end
end

%count the spoof data in training
train_spoof_method_num = {};
train_spoof_method_num_sum = 0;
for k=1:length(Spoof_Method_train)
    dir_spoof_train_method = strcat('./ld-2015_data/Training/',dataSetName,'/Fake/',Spoof_Method_train{k});
    train_spoof_num = length(dir([dir_spoof_train_method,'/*.bmp']));
    if train_spoof_num == 0
        train_spoof_num = length(dir([dir_spoof_train_method,'/*.png']));
    end
    train_spoof_method_num{k} = train_spoof_num;
    train_spoof_method_num_sum = train_spoof_method_num_sum + train_spoof_num;
end

%count the spoof data in testing
test_spoof_method_num = {};
test_spoof_method_num_sum = 0;
for k=1:length(Spoof_Method_test)
    dir_spoof_test_method = strcat('./ld-2015_data/Testing/',dataSetName,'/Fake/',Spoof_Method_test{k});
    test_spoof_num = length(dir([dir_spoof_test_method,'/*.bmp']));
    if test_spoof_num == 0
        test_spoof_num = length(dir([dir_spoof_test_method,'/*.png']));
    end
    test_spoof_method_num{k} = test_spoof_num;
    test_spoof_method_num_sum = test_spoof_method_num_sum + test_spoof_num;
end


switch(methodName)
    case '2015_BSIF'
        cd 2015_BSIF;
        % load filter
        filename=['./texturefilters/ICAtextureFilters_9x9_12bit'];
        load(filename, 'ICAtextureFilters');
        
        Value_Real_Training = zeros(train_real_num,4096);
        Value_Real_Testing = zeros(test_real_num,4096);
        Value_Spoof_Training = zeros(train_spoof_method_num_sum,4096);
        Value_Spoof_Testing = zeros(test_spoof_method_num_sum,4096);
    case '2015_LPQ'
        cd 2015_LPQ;
        Value_Real_Training = zeros(train_real_num,256);
        Value_Real_Testing = zeros(test_real_num,256);
        Value_Spoof_Training = zeros(train_spoof_method_num_sum,256);
        Value_Spoof_Testing = zeros(test_spoof_method_num_sum,256);
    case '2015_WLD'
        cd 2015_WLD
        Value_Real_Training = zeros(train_real_num,960);
        Value_Real_Testing = zeros(test_real_num,960);
        Value_Spoof_Training = zeros(train_spoof_method_num_sum,960);
        Value_Spoof_Testing = zeros(test_spoof_method_num_sum,960);
end

% extract train real
imgNum = 1;
for i=1:train_real_num
    % load image
    if strcmp(collectorName,'DigPerson') == 1 ||strcmp(collectorName,'GreenBit') == 1  
        try
        imgRealTrain = imread(strcat('.',train_real_dir,'/2015_',collectorName,'_Real_',num2str(i),'.png'));
        catch ME
            fprintf('wrong image train real')
            imgNum
        end
        if(size(imgRealTrain,3)>1)
            imgRealTrain = rgb2gray(imgRealTrain);
        end
    else
         imgRealTrain = imread(strcat('.',train_real_dir,'/2015_',collectorName,'_Real_',num2str(i),'.bmp'));
        if(size(imgRealTrain,3)>1)
            imgRealTrain = rgb2gray(imgRealTrain);
        end        
        
    end
    % extract features
    switch(methodName)
        case '2015_BSIF'
            histTrainReal = bsif(imgRealTrain ,ICAtextureFilters,'h');
        case '2015_LPQ'
            histTrainReal = lpq(imgRealTrain,3,1,1,'h');
        case '2015_WLD'
            histTrainReal=WLD_new(imgRealTrain,3,8);
    end
    % store the feature
    Value_Real_Training(imgNum,:)=histTrainReal;
    imgNum=imgNum+1;
end

% extract test real
imgNum = 1;
for i=1:test_real_num
    % load image
    if strcmp(collectorName,'DigPerson') == 1 ||strcmp(collectorName,'GreenBit') == 1 
        try
        imgRealTest = imread(strcat('.',test_real_dir,'/2015_',collectorName,'_Real_',num2str(i),'.png'));
        catch ME
            fprintf('wrong image test real')
            imgNum
        end
        if(size(imgRealTest,3)>1)
            imgRealTest = rgb2gray(imgRealTest);
        end

    else
         imgRealTest = imread(strcat('.',test_real_dir,'/2015_',collectorName,'_Real_',num2str(i),'.bmp'));
        if(size(imgRealTest,3)>1)
            imgRealTest = rgb2gray(imgRealTest);
        end       
        
    end
    % extract features
    switch(methodName)
        case '2015_BSIF'
            histTestReal = bsif(imgRealTest,ICAtextureFilters,'h');
        case '2015_LPQ'
            histTestReal = lpq(imgRealTest,3,1,1,'h');
        case '2015_WLD'
            histTestReal=WLD_new(imgRealTest,3,8);
    end
    % store the feature
    Value_Real_Testing(imgNum,:)=histTestReal;
    imgNum=imgNum+1;
end




% extract train spoof
imgNum = 1;
for k=1:length(Spoof_Method_train)
    for i=1:train_spoof_method_num{k}
        % load image
        if strcmp(collectorName,'DigPerson') == 1 ||strcmp(collectorName,'GreenBit') == 1
            imgSpoofTrain=imread(strcat('.',train_Spoof_dir,'/',Spoof_Method_train{k},'/2015_',collectorName,'_Spoof_',num2str(i),'.png'));
            if(size(imgSpoofTrain,3)>1)
                imgSpoofTrain = rgb2gray(imgSpoofTrain);
            end
        else
            imgSpoofTrain=imread(strcat('.',train_Spoof_dir,'/',Spoof_Method_train{k},'/2015_',collectorName,'_Spoof_',num2str(i),'.bmp'));
            if(size(imgSpoofTrain,3)>1)
                imgSpoofTrain = rgb2gray(imgSpoofTrain);
            end
        end
        
        % extract features
        switch(methodName)
            case '2015_BSIF'
                histTrainReal = bsif(imgSpoofTrain ,ICAtextureFilters,'h');
            case '2015_LPQ'
                histTrainReal = lpq(imgSpoofTrain,3,1,1,'h');
            case '2015_WLD'
                histTrainReal=WLD_new(imgSpoofTrain,3,8);
        end
        % store the feature
        Value_Spoof_Training(imgNum,:)=histTrainReal;
        imgNum=imgNum+1;
        
    end
end

% extract Test spoof
imgNum = 1;
for k=1:length(Spoof_Method_test)
    for i=1:test_spoof_method_num{k}
        % load image
        if strcmp(collectorName,'DigPerson') == 1 ||strcmp(collectorName,'GreenBit') == 1
            imgSpoofTest=imread(strcat('.',test_Spoof_dir,'/',Spoof_Method_test{k},'/2015_',collectorName,'_Spoof_',num2str(i),'.png'));
            if(size(imgSpoofTest,3)>1)
                imgSpoofTest = rgb2gray(imgSpoofTest);
            end
        else
            imgSpoofTest=imread(strcat('.',test_Spoof_dir,'/',Spoof_Method_test{k},'/2015_',collectorName,'_Spoof_',num2str(i),'.bmp'));
            if(size(imgSpoofTest,3)>1)
                imgSpoofTest = rgb2gray(imgSpoofTest);
            end
        end
        
        % extract features
        switch(methodName)
            case '2015_BSIF'
                histTestReal = bsif(imgSpoofTest ,ICAtextureFilters,'h');
            case '2015_LPQ'
                histTestReal = lpq(imgSpoofTest,3,1,1,'h');
            case '2015_WLD'
                histTestReal=WLD_new(imgSpoofTest,3,8);
        end
        % store the feature
        Value_Spoof_Testing(imgNum,:)=histTestReal;
        imgNum=imgNum+1;
        
    end
end


cd ..
if strcmp(methodName,'2015_BSIF') == 1
    nameTrainReal = strcat('Data_',methodName,'_7_12_motion_','Train_Real_',collectorName);
    eval([nameTrainReal,'=Value_Real_Training;']);
    save(strcat('./',methodName,'/Data_',methodName,'_7_12_motion_','Train_Real_',collectorName),nameTrainReal);
    
    nameTestReal = strcat('Data_',methodName,'_7_12_motion_','Test_Real_',collectorName);
    eval([nameTestReal,'=Value_Real_Testing;']);
    save(strcat('./',methodName,'/Data_',methodName,'_7_12_motion_','Test_Real_',collectorName),nameTestReal);
    
    nameTrainSpoof = strcat('Data_',methodName,'_7_12_motion_','Train_Spoof_',collectorName);
    eval([nameTrainSpoof,'=Value_Spoof_Training;']);
    save(strcat('./',methodName,'/Data_',methodName,'_7_12_motion_','Train_Spoof_',collectorName),nameTrainSpoof);
    
    nameTestSpoof = strcat('Data_',methodName,'_7_12_motion_','Test_Spoof_',collectorName);
    eval([nameTestSpoof,'=Value_Spoof_Testing;']);
    save(strcat('./',methodName,'/Data_',methodName,'_7_12_motion_','Test_Spoof_',collectorName),nameTestSpoof);
    
elseif strcmp(methodName,'2015_LPQ') == 1
    nameTrainReal = strcat('Data_',methodName,'_3_11_motion_','Train_Real_',collectorName);
    eval([nameTrainReal,'=Value_Real_Training;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_11_motion_','Train_Real_',collectorName),nameTrainReal);
    
    nameTestReal = strcat('Data_',methodName,'_3_11_motion_','Test_Real_',collectorName);
    eval([nameTestReal,'=Value_Real_Testing;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_11_motion_','Test_Real_',collectorName),nameTestReal);
    
    nameTrainSpoof = strcat('Data_',methodName,'_3_11_motion_','Train_Spoof_',collectorName);
    eval([nameTrainSpoof,'=Value_Spoof_Training;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_11_motion_','Train_Spoof_',collectorName),nameTrainSpoof);
    
    nameTestSpoof = strcat('Data_',methodName,'_3_11_motion_','Test_Spoof_',collectorName);
    eval([nameTestSpoof,'=Value_Spoof_Testing;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_11_motion_','Test_Spoof_',collectorName),nameTestSpoof);
    
elseif strcmp(methodName,'2015_WLD') == 1
    nameTrainReal = strcat('Data_',methodName,'_3_8_motion_','Train_Real_',collectorName);
    eval([nameTrainReal,'=Value_Real_Training;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_8_motion_','Train_Real_',collectorName),nameTrainReal);
    
    nameTestReal = strcat('Data_',methodName,'_3_8_motion_','Test_Real_',collectorName);
    eval([nameTestReal,'=Value_Real_Testing;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_8_motion_','Test_Real_',collectorName),nameTestReal);
    
    nameTrainSpoof = strcat('Data_',methodName,'_3_8_motion_','Train_Spoof_',collectorName);
    eval([nameTrainSpoof,'=Value_Spoof_Training;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_8_motion_','Train_Spoof_',collectorName),nameTrainSpoof);
    
    nameTestSpoof = strcat('Data_',methodName,'_3_8_motion_','Test_Spoof_',collectorName);
    eval([nameTestSpoof,'=Value_Spoof_Testing;']);
    save(strcat('./',methodName,'/Data_',methodName,'_3_8_motion_','Test_Spoof_',collectorName),nameTestSpoof);
end


training_data = [Value_Real_Training;Value_Spoof_Training];
testing_data = [Value_Real_Testing;Value_Spoof_Testing];
real_label = ones(train_real_num,1);
fake_label = zeros(train_spoof_method_num_sum,1);
training_label = [real_label;fake_label];
testing_label = training_label;
SVM_model = svmtrain(training_data,training_label);
Predict_Real = svmclassify(SVM_model,Value_Real_Testing);
correct_Real = sum(Predict_Real);
Predict_Spoof = svmclassify(SVM_model,Value_Spoof_Testing);
correct_Spoof = test_spoof_method_num_sum-sum(Predict_Spoof);
acc = (correct_Real+correct_Spoof)/(test_spoof_method_num_sum+test_real_num)