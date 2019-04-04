% EXAMPLE 1
% Load image
files = dir('/images/*.png');

% Parameters:
clear param
param.imageSize = [512 512]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;


for col=0:4
    for row=0:5
        string = strcat('./images/human_row=', num2str(row),'_col=', num2str(col), '.png');
        image = imread(string);
        [gist1, param] = LMgist(image, '', param);

        % Visualization
        figure
        %subplot(121)
        %imshow(image)
        %title('Input image')
        %subplot(122)
        showGist(gist1, param)
        title('Descriptor')
        saveas(gcf, strcat('./images/human_row=', num2str(row),'_col=', num2str(col), '_gist'),'png');
    
    end
end

for col=0:3
    for row=0:5
        string = strcat('./images/nohuman_row=', num2str(row),'_col=', num2str(col), '.png');
        image = imread(string);
        [gist1, param] = LMgist(image, '', param);

        % Visualization
        figure
        %subplot(121)
        %imshow(image)
        %title('Input image')
        %subplot(122)
        showGist(gist1, param)
        title('Descriptor')
        saveas(gcf, strcat('./images/nohuman_row=', num2str(row),'_col=', num2str(col), '_gist'),'png');
    
    end
end

ff

for i=1:length(files)
    image = eval(['load ' files(i).name ' -ascii']);
    [gist1, param] = LMgist(image, '', param);

    % Visualization
    figure
    subplot(121)
    imshow(img1)
    title('Input image')
    subplot(122)
    showGist(gist1, param)
    title('Descriptor')

end