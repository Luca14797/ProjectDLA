function layers = new_architecture(inputSize, version)

    if version == 1 %8

        layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(3,32)
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,64)
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,128)
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,256)
        reluLayer
        
        fullyConnectedLayer(1024)
        reluLayer
        
        fullyConnectedLayer(1024)
        reluLayer

        fullyConnectedLayer(4)
        softmaxLayer
        classificationLayer];

    elseif version == 2 %9

        layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(5,32)
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,64)
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,128)
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,256)
        reluLayer
        
        fullyConnectedLayer(1024)
        reluLayer
        
        fullyConnectedLayer(1024)
        reluLayer

        fullyConnectedLayer(4)
        softmaxLayer
        classificationLayer];
    
    elseif version == 3 %11
        
        layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(7,32)
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(7,64)
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(5,128)
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,256)
        reluLayer
        
        fullyConnectedLayer(1024)
        reluLayer
        
        fullyConnectedLayer(1024)
        reluLayer

        fullyConnectedLayer(4)
        softmaxLayer
        classificationLayer];

    end

end
