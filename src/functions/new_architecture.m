function layers = new_architecture(inputSize, version)

    if version == 1 %8

        layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(3,32)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,64)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,128)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,256)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        fullyConnectedLayer(4)
        batchNormalizationLayer
        softmaxLayer
        classificationLayer];

    elseif version == 2 %9

        layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(5,32)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,64)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,128)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,256)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        fullyConnectedLayer(4)
        batchNormalizationLayer
        softmaxLayer
        classificationLayer];
    
    elseif version == 3 %11
        
        layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(7,32)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(7,64)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,128)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,256)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        leakyReluLayer(0.01)
        
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        leakyReluLayer(0.01)

        fullyConnectedLayer(4)
        batchNormalizationLayer
        softmaxLayer
        classificationLayer];
    
    end

end
