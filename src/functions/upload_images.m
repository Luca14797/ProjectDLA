function data = upload_images(directory)
    
    last = 0;

    for i = 1 : size(directory)
        
        files = dir(directory{i});

        %remove '..' and '.' directories
        files(~cellfun(@isempty, regexp({files.name}, '\.*')))=[];

        filePattern = {};

        for k = 1 : length(files)

            pattern = fullfile(directory{i}, num2str(k));

            last = last + 1;

            filePattern{last} = pattern;

        end
        
    end
    
    data = imageDatastore(filePattern, 'FileExtensions', '.png', 'LabelSource', 'foldernames');
    
end
