% const
STD = 1;
NORM = 2;

% PATHS
basepath = '..';

if ispc
    functions_folder = fullfile(basepath, 'src\functions');
else
    functions_folder = fullfile(basepath, 'src/functions'); 
end

if ispc
    svm_folder = fullfile(basepath, 'src\SVMlinear');
else
    svm_folder = fullfile(basepath, 'src/SVMlinear'); 
end

addpath(functions_folder)
addpath(svm_folder)

% DATASET
dataset_dir = 'dataset';

if ispc
    data_dir = 'src\data';
else
    data_dir = 'src/data';
end

% FLAGS
upload_dataset = 0;
show_img = 0;
do_alexnet = 0;
do_resnet18 = 1;
do_vgg16 = 0;
do_googlenet = 0;
do_svm = 1;
do_fine_tuning = 0;
do_new_architecture = 0;
include_segm = 0;

% VARIABLES
file_train = 'train.mat';
file_validation = 'validation.mat';
file_test = 'test.mat';

% CLASSES
classes = {"Normal Pollen", "Anomalus Pollen", "Alnus", "Debris"};

rng('default');
rng(0);

gpurng('default');
gpurng(0);

%% UPLOAD IMAGES

if (upload_dataset == 1)
    
    fprintf('Upload images and create dataset ...\n');
    
    % Upload train and validation images
    filePatternObj = fullfile(basepath, dataset_dir, 'train', 'images', 'train_OBJ');
    filePatternSegm = fullfile(basepath, dataset_dir, 'train', 'images', 'train_SEGM');
    
    filePattern = {filePatternObj};
    
    if include_segm
        
        filePattern = {filePatternObj, filePatternSegm};
    
    end
    
    train = upload_images(filePattern);
    [train, validation] = splitEachLabel(train, 0.8, 'randomized');
    save(fullfile(basepath, data_dir, file_train), 'train', '-v7.3');
    save(fullfile(basepath, data_dir, file_validation), 'validation', '-v7.3');

    % Upload test images
    filePattern = {fullfile(basepath, dataset_dir, 'test', 'images')};
    test = upload_images(filePattern);
    save(fullfile(basepath, data_dir, file_test), 'test', '-v7.3');
    
else
    
    fprintf('Load dataset ...\n');
    
    % Load train data
    load(fullfile(basepath, data_dir, file_train));
    
    % Load validation data
    load(fullfile(basepath, data_dir, file_validation));
    
    % Load test data
    load(fullfile(basepath, data_dir, file_test));
    
end



%% PRE PROCESSING

fprintf('Pre processing ...\n');

imageAugmenter = imageDataAugmenter('RandRotation',@() -180 + 360*rand, ... 
                                    'RandXReflection', true, ...
                                    'RandXScale',[1, 1.05], ...
                                    'RandYReflection', true, ...
                                    'RandYScale',[1, 1.05]);
augTrain = augmentedImageDatastore([84 84 3], train, 'DataAugmentation', imageAugmenter);


%%
if (do_alexnet == 1)

    fprintf('Load Alexnet ...\n');

    net = alexnet;

    inputSize = net.Layers(1).InputSize;

elseif (do_resnet18 == 1)

    fprintf('Load Resnet18 ...\n');

    net = resnet18;

    inputSize = net.Layers(1).InputSize;

elseif (do_vgg16 == 1)

    fprintf('Load VGG16 ...\n');

    net = vgg16;

    inputSize = net.Layers(1).InputSize;
    
elseif (do_googlenet == 1)

    fprintf('Load GoogleNet ...\n');

    net = googlenet;

    inputSize = net.Layers(1).InputSize;


end 
%%

if (do_svm == 1)
    
    fprintf('Augmenting images ...\n');

    trainAug = augmentedImageDatastore(inputSize(1:2), train);
    testAug = augmentedImageDatastore(inputSize(1:2), test);
    
    [accuracy, YPred, YTest] = svm_classification(net, train, test, trainAug, testAug, show_img);
    
elseif (do_fine_tuning == 1)
    
    [accuracy, YPred, YTest] = fine_tuning(net, 'vgg16', train, validation, inputSize);
    %ResNet18 = 'fc1000' and 'ClassificationLayer_predictions'
    %GoogleNet = 'loss3-classifier' and 'output'
    
elseif (do_new_architecture == 1)
    
   %accuracies = grid_search(train, [0.01, 0.001, 0.0001], ["adam", "rmsprop", "sgdm"], [16, 32, 64], [1, 2]);
   meanTest = 0;
   meanVal = 0;
   meanTrain = 0;
   for i=1:3
        [accuracyTest, accuracyVal, accuracyTrain, YPred, YTest] = train_new_architecture(train, validation, test, augTrain, 2);
        meanTest = meanTest + accuracyTest;
        meanVal = meanVal + accuracyVal;
        meanTrain = meanTrain + accuracyTrain;
        fprintf('Accuracy #%d Test set: %d\n', i, accuracyTest);
        fprintf('Accuracy #%d Validation set: %d\n', i, accuracyVal);
        fprintf('Accuracy #%d Train set: %d\n', i, accuracyTrain);
        
   end
   
   fprintf('Mean accuracy Test set: %d\n', (meanTest/3));
   fprintf('Mean accuracy Validation set: %d\n', (meanVal/3));
   fprintf('Mean accuracy Train set: %d\n', (meanTrain/3));
    
end

cm = confusionchart(YTest ,YPred);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';  
cm.NormalizedValues
cm

show_results(classes, test, grp2idx(YTest), grp2idx(YPred));
