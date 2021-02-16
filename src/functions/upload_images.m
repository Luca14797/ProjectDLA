function data = upload_images(directory)

    files = dir(directory);

    %remove '..' and '.' directories
    files(~cellfun(@isempty, regexp({files.name}, '\.*')))=[];

    filePattern = {};
    
    last = 0;
    
    
    for k = 1 : length(files)
        
        pattern = fullfile(directory, num2str(k));
        
        last = last+1;
        
        filePattern{last} = pattern;
        
    end
    
    data = imageDatastore(filePattern,'FileExtensions','.png','LabelSource','foldernames');

    
end
