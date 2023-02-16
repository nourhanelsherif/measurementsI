%% PART 2 Q7, radius is 50
C = imread('IMG-3609.jpg');

figure(1)
I = mean(C,3);
Icrop=I(200:1200,200:1200);
imagesc(I);
colormap(gray)

%column 1 of subplot
figure(2);
subplot(2,3,1);
imagesc(Icrop);
colormap(subplot(2,3,1),'gray');
axis image;
title('Original Image');
xlabel('X position (pixels)');
ylabel('Y position (pixels)');

s4=fft2(Icrop);
s4plot=fftshift(abs(s4));
subplot(2,3,4);
lim=linspace(-0.5, 0.5,length(s4plot));
imagesc(lim, lim, s4plot);
colorbar;
caxis([0, 2e5]);
title('Original FFT');
xlabel('X frequency (cycles/pix)');
ylabel('Y frequency (cycles/pix)');

%column 2 of subplot
%new radius!
%make LPF
radius = 50;
[X,Y]=meshgrid(1:size(Icrop,1), 1:size(Icrop,2));
X=X-mean(X(:));
Y=Y-mean(Y(:));
mask=sqrt(X.^2+Y.^2)<radius;

%checking if the abs value of the masked fft is correct
s5=mask.*fftshift(s4);
subplot(2,3,5);
lim=linspace(-0.5, 0.5,length(s5));
imagesc(lim, lim, abs(s5));
colorbar;
caxis([0, 2e5]);
title('FFT with high frequencies removed');
xlabel('X frequency (cycles/pix)');
ylabel('Y frequency (cycles/pix)');

%shift masked fft
s5shift=fftshift(s5);

%inverse fft to get image
s2=abs(ifft2(s5shift));
subplot(2,3,2);
imagesc(s2);
colormap(subplot(2,3,2),'gray');
axis image;
title('Low-pass Filtered Image');
xlabel('X position (pixels)');
ylabel('Y position (pixels)');

%column 3 of subplot
%make HPF
radius = 50;
[X,Y]=meshgrid(1:size(Icrop,1), 1:size(Icrop,2));
X=X-mean(X(:));
Y=Y-mean(Y(:));
mask2=sqrt(X.^2+Y.^2)>radius;

s6=mask2.*fftshift(s4);
subplot(2,3,6);
lim=linspace(-0.5, 0.5,length(s6));
imagesc(lim, lim, abs(s6));
colorbar;
caxis([0, 2e5]);
title('FFT with low frequencies removed');
xlabel('X frequency (cycles/pix)');
ylabel('Y frequency (cycles/pix)');

s6shift=fftshift(s6);

%inverse fft to get image
s3=abs(ifft2(s6shift));
subplot(2,3,3);
imagesc(s3);
colormap(subplot(2,3,3),'gray');
axis image;
title('High-pass Filtered Image');
xlabel('X position (pixels)');
ylabel('Y position (pixels)');