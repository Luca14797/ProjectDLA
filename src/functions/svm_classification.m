function accuracy = svm_classification(net, train, test, trainAug, testAug, show_img)

    fprintf('Activations ...\n');

    layer = 'pool5';
    featuresTrain = activations(net, trainAug, layer, 'OutputAs', 'rows');
    featuresTest = activations(net, testAug, layer, 'OutputAs', 'rows');

    whos featuresTrain

    YTrain = train.Labels;
    YTest = test.Labels;

    fprintf('Fitting SVM ...\n');

    classifier = fitcecoc(featuresTrain, YTrain);

    fprintf('Prediction ...\n');

    YPred = predict(classifier, featuresTest);

    if (show_img == 1)
        
        idx = [1 5 10 15];
        figure
        for i = 1:numel(idx)
            
            subplot(2,2,i)
            I = readimage(test, idx(i));
            label = YPred(idx(i));
            imshow(I)
            title(char(label))
            
        end
        
    end

    accuracy = mean(YPred == YTest);
    
end
