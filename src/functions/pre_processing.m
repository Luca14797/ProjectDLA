function [train, test] = pre_processing(train, test, options)
    
    STD = 1;
    NORM = 2;

    if (bitand(options, NORM))
        
        fprintf('Normalization ...\n');
        
        train.X = train.X/255;
        test.X = test.X/255;
        
    end
        
    if (bitand(options, STD))
        
        fprintf('Standardization ...\n');
        
        m_train = mean(train.X);
        s_train = std(train.X);

        train.X = (train.X-m_train)./(s_train+eps);

        test.X = (test.X-m_train)./(s_train+eps);
        
        
    end

end
