function [] = show_results(classes, desc_test, labels_test, labels_result)

    figure;

    for i=1:length(classes)

        ind=find(labels_test==i);
        indcorr=ind(find(labels_result(ind)==labels_test(ind)));
        indmiss=ind(find(labels_result(ind)~=labels_test(ind)));

        clf
        imgcorr={};
        if length(indcorr)
            for j=1:length(indcorr) 
                imgcorr{end+1}=imread(char(desc_test.Files(indcorr(j))));
            end
            subplot(1,2,1), showimage(comb_image(imgcorr,[],1))
            title(sprintf('%d Correctly classified %s images',length(indcorr),classes{i}))
        end

        imgmiss={};
        if length(indmiss)
            for j=1:length(indmiss)
                imgmiss{end+1}=imread(char(desc_test.Files(indmiss(j))));
            end
            subplot(1,2,2), showimage(comb_image(imgmiss,[],1))
            title(sprintf('%d Miss-classified %s images',length(indmiss),classes{i}))
        end

        fprintf('Classe %s, percentuale accuratezza: %1.4f\n', classes{i}, (length(indcorr)/(length(indcorr) + length(indmiss)))*100);

        pause;
    end
    
end
