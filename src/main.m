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
do_resnet18 = 0;
do_vgg16 = 0;
do_svm = 0;
do_fine_tuning = 0;
do_new_architecture = 1;

% VARIABLES
file_train = 'train.mat';
file_test = 'test.mat';

rng(0);

%% UPLOAD IMAGES

if (upload_dataset == 1)
    
    fprintf('Upload images and create dataset ...\n');
    
    % Upload train images
    filePattern = fullfile(fullfile(basepath, dataset_dir, 'train', 'images', 'train_OBJ'));
    train = upload_images(filePattern);
    save(fullfile(basepath, data_dir, file_train), 'train', '-v7.3');

    % Upload test images
    filePattern = fullfile(fullfile(basepath, dataset_dir, 'test', 'images'));
    test = upload_images(filePattern);
    save(fullfile(basepath, data_dir, file_test), 'test', '-v7.3');
    
else
    
    fprintf('Load dataset ...\n');
    
    % Load train data
    load(fullfile(basepath, data_dir, file_train));
    
    % Load test data
    load(fullfile(basepath, data_dir, file_test));
    
end


%{
%% PRE PROCESSING

fprintf('Pre processing ...\n');

[train, test] = pre_processing(train, test, NORM+STD);
%}

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

end 
%%

if (do_svm == 1)
    
    fprintf('Augmenting images ...\n');

    trainAug = augmentedImageDatastore(inputSize(1:2), train);
    testAug = augmentedImageDatastore(inputSize(1:2), test);
    
    accuracy = svm_classification(net, train, test, trainAug, testAug, show_img);
    
elseif (do_fine_tuning == 1)
    
    accuracy = fine_tuning(net, 'alexnet', train, inputSize);

elseif (do_new_architecture == 1)
    
   accuracies = grid_search(train, [0.01, 0.001, 0.0001], ["adam", "rmsprop", "sgdm"], [16, 32, 64], [1, 2]);
    
end
