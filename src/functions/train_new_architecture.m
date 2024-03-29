function [accuracyTest, accuracyVal, accuracyTrain, YPred, YTest] = train_new_architecture(train, validation, test, augTrain, version)

    layers = new_architecture([84 84 3], version);
    options = trainingOptions('adam', 'MiniBatchSize',32, ...
        'InitialLearnRate',0.0003, 'MaxEpochs', 10, ...
        'Shuffle','every-epoch', 'ValidationData', validation, ...
        'ValidationFrequency',281, 'Verbose',true, ...
        'Plots','training-progress')
        %%'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.90, 'LearnRateDropPeriod', 10);
    net = trainNetwork(augTrain, layers, options);
    
    YPred = classify(net, validation);
    YVal = validation.Labels;

    accuracyVal = sum(YPred == YVal)/numel(YVal);
    
    YPred = classify(net, train);
    YTrain = train.Labels;

    accuracyTrain = sum(YPred == YTrain)/numel(YTrain);
    
    YPred = classify(net, test);
    YTest = test.Labels;

    accuracyTest = sum(YPred == YTest)/numel(YTest);
    
end
