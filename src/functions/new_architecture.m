function layers = new_architecture(inputSize, version)

    if version == 1

    layers = [
        imageInputLayer(inputSize)

        convolution2dLayer(3,32,'Padding','same')
        leakyReluLayer(0.1)

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,64,'Padding','same')
        leakyReluLayer(0.1)

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,128,'Padding','same')
        leakyReluLayer(0.1)
        
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,256,'Padding','same')
        leakyReluLayer(0.1)
        
        fullyConnectedLayer(256)
        leakyReluLayer(0.1)
        fullyConnectedLayer(4)
        softmaxLayer
        classificationLayer];

    elseif version == 2

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

    end

end
