function [gray, rgb, cNum] = linkage_cluster(n, I, X, pdistMode, linkageMode, cutoff)
    
    Y = pdist(X, pdistMode);
    Z = linkage(Y, linkageMode);
    T = cluster(Z,'cutoff',cutoff);
    cNum = max(max(T));
    newX = uint8(255 * mat2gray(T));
    C = unique(newX);

    %%make it colorful
    newColor = 255*rand(size(C,1),3);
    newColor = uint8(newColor);
    colorImage = X(:,1);
    for i=1:size(C,1)
        colorImage(find(newX == C(i,1)),1) = newColor(i,1);  
        colorImage(find(newX == C(i,1)),2) = newColor(i,2);  
        colorImage(find(newX == C(i,1)),3) = newColor(i,3);  
    end

    grayImage = reshape(newX ,[n n]);
    i1 = reshape(colorImage(:,1) ,[n n]);
    i2 = reshape(colorImage(:,2) ,[n n]);
    i3 = reshape(colorImage(:,3) ,[n n]);
    I(:,:,1) = i1;
    I(:,:,2) = i2;
    I(:,:,3) = i3;
    
    gray = grayImage;
    rgb = I;

end