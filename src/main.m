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

addpath(functions_folder)

% DATASET
dataset_dir = 'dataset';

if ispc
    data_dir = 'src\data';
else
    data_dir = 'src/data';
end

% FLAGS
upload_dataset = 0;

% VARIABLES
file_train = 'train.mat';
file_test = 'test.mat';

%% UPLOAD IMAGES



if (upload_dataset == 1)
    
    fprintf('Upload images and create dataset ...\n');
    
    % Upload train images
    filePattern = fullfile(fullfile(basepath, dataset_dir, 'train', 'images'));
    train = upload_train_images(filePattern);
    save(fullfile(basepath, data_dir, file_train), 'train', '-v7.3');

    % Upload test images
    filePattern = fullfile(fullfile(basepath, dataset_dir, 'test', 'images'));
    test = upload_test_images(filePattern);
    save(fullfile(basepath, data_dir, file_test), 'test', '-v7.3');
    
else
    
    fprintf('Load dataset ...\n');
    
    % Load train data
    load(fullfile(basepath, data_dir, file_train));
    
    % Load test data
    load(fullfile(basepath, data_dir, file_test));
    
end

%% PRE PROCESSING

fprintf('Pre processing ...\n');

[train, test] = pre_processing(train, test, NORM+STD);

%% ALEX NET

net = alexnet;

inputSize = net.Layers(1).InputSize;

train = augmentedImageDatastore(inputSize(1:2), train);
test = augmentedImageDatastore(inputSize(1:2), test);

size(test.X)
size(train.X)

