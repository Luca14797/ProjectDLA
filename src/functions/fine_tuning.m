function accuracy = fine_tuning(net, net_name, train, validation, inputSize, level_name, newLevel_name)
    
    if (net_name == "resnet18" || net_name == "googlenet")
        
        % DAG network
        
        lgraph = layerGraph(net);
        numClasses = numel(categories(train.Labels));
        newLearnableLayer = fullyConnectedLayer(numClasses, 'Name','new_fc1000', ...
            'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
        lgraph = replaceLayer(lgraph,level_name,newLearnableLayer);
        newClassLayer = classificationLayer('Name','new_classoutput');
        lgraph = replaceLayer(lgraph,newLevel_name,newClassLayer);
        
    elseif (net_name == "alexnet" || net_name == "vgg16")
        
        % Series network
        
        layersTransfer = net.Layers(1:end-3);
        numClasses = numel(categories(train.Labels));
        lgraph = [
            layersTransfer
            fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
            softmaxLayer
            classificationLayer];
        
    end
    
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter('RandXReflection',true, ...
        'RandXTranslation',pixelRange, 'RandYTranslation',pixelRange);
    trainAug = augmentedImageDatastore(inputSize(1:2),train, ...
        'DataAugmentation',imageAugmenter);
    validationAug = augmentedImageDatastore(inputSize(1:2),validation);
    
    options = trainingOptions('sgdm', 'MiniBatchSize',16, 'MaxEpochs',5, ...
        'InitialLearnRate',1e-4, 'Shuffle','every-epoch', ...
        'ValidationData',validationAug, 'ValidationFrequency', 281, ...
        'Verbose',false, 'Plots','training-progress');

    netTransfer = trainNetwork(trainAug,lgraph,options);
    
    [YPred,scores] = classify(netTransfer,validationAug);
    
    YValidation = validation.Labels;
    accuracy = mean(YPred == YValidation);

end
