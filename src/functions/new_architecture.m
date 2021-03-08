function layers = new_architecture(inputSize, version)

if version == 1
    
layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,16,'Padding','same')
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

elseif version == 2
    
    layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,8,'Padding','same')
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];
    
end
    


end
