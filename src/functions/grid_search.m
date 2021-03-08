function accuracies = grid_search(train, learning_rates, optimizers, batch_sizes, versions)

[train,validation] = splitEachLabel(train,0.8,'randomized');

accuracies = zeros(length(learning_rates), length(optimizers), length(batch_sizes), length(versions));

fprintf('Total tests: %d\n', numel(accuracies));

for lr=1:length(learning_rates)
    
    for o=1:length(optimizers)
        
        for bs=1:length(batch_sizes)
            
            for v=1:length(versions)

                layers = new_architecture([84 84 3], versions(v));
                
                
                options = trainingOptions(optimizers(o), ...
                    'MiniBatchSize',batch_sizes(bs), ...
                    'InitialLearnRate',learning_rates(lr), ...
                    'MaxEpochs',5, ...
                    'Shuffle','every-epoch', ...
                    'ValidationData',validation, ...
                    'ValidationFrequency',30, ...
                    'Verbose',true, ...
                    'Plots','none');
                
                net = trainNetwork(train,layers,options);

                YPred = classify(net,validation);
                YValidation = validation.Labels;

                accuracies(lr, o, bs, v) = sum(YPred == YValidation)/numel(YValidation);
                
            end
            
        end

    end

end

[m,i] = max(accuracies, 'all');

fprintf('Best configuration: %d', m);
fprintf('Learning rate: %d', leaning_rates(i(1)));
fprintf('Optimizer: %d', optimizers(i(2)));
fprintf('Batch size: %d', batch_sizes(i(3)));
fprintf('Version: %d', versions(i(4)));


end