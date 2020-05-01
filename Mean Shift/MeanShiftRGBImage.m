clc;clear;close all

x = imread("akbari.jpg");
x = imresize(x,[400 400]);

figure;
imshow(x);

x = double(x); 
Sigma=54;
Iter=15;
n=length(x);

Ceof=1/(sqrt(2*pi)*n*Sigma);
l = 1;
R=zeros(1,length(x ),3);
for i=1:length(x)
    z=x(i,l,:);
    for k=1:Iter
        y = (z-x).^2;
        y = y(:,:,1) + y(:,:,2) + y(:,:,3);
        yt=y/(2*Sigma^2);
        a = sum(x.*exp(-yt));
        b = sum(exp(-yt));
        z(:,:,1)=a(:,:,1)/b;
        z(:,:,2)=a(:,:,2)/b;
        z(:,:,3)=a(:,:,3)/b;
    end
    R(1,i,:)=z;
end
Threshold = ones(1,1,3) * 20;
FF(1,1,:)=R(1,1,:);

b = abs(diff(sort(R)));
c = b > Threshold;
d = find(c);
d = d(find(d < size(x,1)));
e = b(1,d,:);
FF=[FF e];

g = x - FF(1,1,:);
j = g;

for i=2:size(FF,2)
    h = x - FF(1,i,:);
    j = min(g,h);
end

figure(1);imagesc(j)