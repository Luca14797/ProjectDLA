function [accuracyTest, accuracyVal, accuracyTrain] = train_new_architecture(train, validation, test, augTrain, augValidation, version)

    layers = new_architecture([84 84 3], version);
    options = trainingOptions('adam', 'MiniBatchSize',32, ...
        'InitialLearnRate',0.0003, 'MaxEpochs',5, ...
        'Shuffle','every-epoch', 'ValidationData',augValidation, ...
        'ValidationFrequency',281, 'Verbose',true, ...
        'Plots','training-progress');

    % , 'L2Regularization',0.005
    % 'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.90, 'LearnRateDropPeriod',1,
    net = trainNetwork(augTrain, layers, options);

    YPred = classify(net, test);
    YTest = test.Labels;

    accuracyTest = sum(YPred == YTest)/numel(YTest);
    
    YPred = classify(net, validation);
    YVal = validation.Labels;

    accuracyVal = sum(YPred == YVal)/numel(YVal);
    
    YPred = classify(net, train);
    YTrain = train.Labels;

    accuracyTrain = sum(YPred == YTrain)/numel(YTrain);
    
end
