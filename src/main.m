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
do_alexnet = 1;
do_resnet18 = 0;
do_vgg16 = 0;

% VARIABLES
file_train = 'train.mat';
file_test = 'test.mat';

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

targetSize = [227, 227, 3];

train = transform(train, @(x) imresize(x, targetSize));
trainAug = train.UnderlyingDatastore;

test = transform(test, @(x) imresize(x, targetSize));
testAug = test.UnderlyingDatastore;

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
%{
fprintf('Augmenting images ...\n');

trainAug = augmentedImageDatastore(inputSize(1:2), train);
testAug = augmentedImageDatastore(inputSize(1:2), test);
%}
fprintf('Activations ...\n');

layer = 'pool5';
featuresTrain = activations(net,trainAug,layer,'OutputAs','rows');
featuresTest = activations(net,testAug,layer,'OutputAs','rows');

whos featuresTrain

YTrain = train.Labels;
YTest = test.Labels;

fprintf('Fitting SVM ...\n');

classifier = fitcecoc(featuresTrain,YTrain);

fprintf('Prediction ...\n');

YPred = predict(classifier,featuresTest);

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(test,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

accuracy = mean(YPred == YTest);

function dataOut = normalization(data)
    fprintf('1\n');
    dataOut = cell(size(data));
    for col = 1:size(data,2)
        fprintf('2\n');
        for idx = 1:size(data,1)
            fprintf('3\n');
            fprintf('%d',data{idx,col});
            
            temp = single(data{idx,col});
            temp = imresize(temp,[32,32]);
            temp = rescale(temp);
            dataOut{idx,col} = temp;
            
        end
        
    end
    fprintf('4\n');
end
