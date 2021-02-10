function data = upload_test_images(directory)

    files = dir(directory);

    %remove '..' and '.' directories
    files(~cellfun(@isempty, regexp({files.name}, '\.*')))=[];

    data.X = [];
    data.y = [];

    for k = 1 : length(files)

        dirImages = fullfile(directory, num2str(k), 'test_OBJ', '*.png');
        listImages = dir(dirImages);

        X = zeros(length(listImages), 84, 84, 3);
        y = zeros(length(listImages), 1);

        fprintf(1, 'Now reading %s\n', fullfile(directory, num2str(k)));

        for i = 1 : length(listImages)

            baseFileName = listImages(i).name;
            fullFileName = fullfile(listImages(i).folder, baseFileName);
            % Now do whatever you want with this file name,
            % such as reading it in as an image array with imread()
            imageArray = imread(fullFileName);
            X(i,:,:,:) = imageArray;
            y(i) = k;

        end

        data.X = cat(1, data.X, X);
        data.y = cat(1, data.y, y);

    end
    
end
