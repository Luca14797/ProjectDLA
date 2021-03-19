function accuracy = train_new_architecture(train, version)

    [train,validation] = splitEachLabel(train, 0.8, 'randomized');

    layers = new_architecture([84 84 3], version);
    options = trainingOptions('adam', 'MiniBatchSize',32, ...
        'InitialLearnRate',0.0001, 'MaxEpochs',15, ...
        'Shuffle','every-epoch', 'ValidationData',validation, ...
        'ValidationFrequency',281, 'Verbose',true, ...
        'Plots','training-progress', 'L2Regularization',0.005);

    net = trainNetwork(train,layers,options);

    YPred = classify(net,validation);
    YValidation = validation.Labels;

    accuracy = sum(YPred == YValidation)/numel(YValidation);
    
end
