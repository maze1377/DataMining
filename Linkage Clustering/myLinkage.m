
% load fisheriris
% X = meas;
n = 100;

I = imread("process.jpg");
I = imresize(I,[n n]);
windowWidth = 5; % Whatever you want.  More blur for larger numbers.
kernel = ones(windowWidth) / windowWidth ^ 2;
blurredImage = imfilter(I, kernel); % Blur the image.

X = I;
% X = rgb2gray(X);
X1 = X(:,:,1);
X2 = X(:,:,2);
X3 = X(:,:,3);
X1 = reshape(X1 ,[n*n 1]);
X2 = reshape(X2 ,[n*n 1]);
X3 = reshape(X3 ,[n*n 1]);
X = [X1 X2 X3];

for k=1:7
    if k == 1
        %Unweighted average distance (UPGMA)
        linkageMode = 'average';
    elseif k == 2
        %Centroid distance (UPGMC), appropriate for Euclidean distances only
        linkageMode = 'centroid';
        continue
    elseif k == 3
        %Farthest distance
        linkageMode = 'complete';   
    elseif k == 4
        %Weighted center of mass distance (WPGMC), appropriate for Euclidean distances only
        linkageMode = 'median';  
        continue
    elseif k == 5
        %Shortest distance
        linkageMode = 'single';   
    elseif k == 6
        %Inner squared distance (minimum variance algorithm), appropriate for Euclidean distances only
        linkageMode = 'ward'; 
        continue
    elseif k == 7
        %Weighted average distance (WPGMA)
        linkageMode = 'weighted'; 
        continue
    end
    for i=1:12
        if i == 1
            pdistMode = 'euclidean';
        elseif i == 2
            pdistMode = 'squaredeuclidean';
        elseif i == 3
            pdistMode = 'seuclidean';
        elseif i == 4
            pdistMode = 'mahalanobis';
        elseif i == 5
            pdistMode = 'cityblock';
        elseif i == 6
            pdistMode = 'minkowski';
        elseif i == 7
            pdistMode = 'chebychev';
        elseif i == 8
            pdistMode = 'cosine';
        elseif i == 9
            pdistMode = 'correlation';
        elseif i == 10
            pdistMode = 'hamming';
        elseif i == 11
            pdistMode = 'jaccard';
        elseif i == 12
            pdistMode = 'spearman';
        end
        for j=1:5
            if j == 1
                cutoff = 0.1;
            elseif j==2
                cutoff = 0.25;
            elseif j==3
                cutoff = 0.5;
            elseif j==4
                cutoff = 1.0;
            elseif j==5
                cutoff = 5;    
            end
            [gray, rgb, cNum] = linkage_cluster(n, I, X, pdistMode, linkageMode, cutoff);
            m = 240;
            gray = imresize(gray,[m m]);
            rgb = imresize(rgb,[m m]);
            str = pdistMode + "-" + linkageMode + "-" + cutoff + "-" + cNum;
            gray_filename = "res/" + k + "/gray_" + str + ".jpg";
            rgb_filename = "res/" + k + "/rgb_" + str + ".jpg";
            imwrite(gray, gray_filename);
            imwrite(rgb, rgb_filename);
        end
    end
end
