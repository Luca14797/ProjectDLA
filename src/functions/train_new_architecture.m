function [accuracyTest, accuracyVal] = train_new_architecture(train, test, version)

    [train,validation] = splitEachLabel(train, 0.8, 'randomized');

    layers = new_architecture([84 84 3], version);
    options = trainingOptions('adam', 'MiniBatchSize',32, ...
        'InitialLearnRate',0.0003, 'MaxEpochs',5, ...
        'Shuffle','every-epoch', 'ValidationData',validation, ...
        'ValidationFrequency',281, 'Verbose',true, ...
        'Plots','training-progress');

    % , 'L2Regularization',0.005
    % 'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.90, 'LearnRateDropPeriod',1,
    net = trainNetwork(train, layers, options);

    YPred = classify(net, test);
    YTest = test.Labels;

    accuracyTest = sum(YPred == YTest)/numel(YTest);
    
    YPred = classify(net, validation);
    YVal = validation.Labels;

    accuracyVal = sum(YPred == YVal)/numel(YVal);
    
end
