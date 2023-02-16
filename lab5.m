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
%radius of 10 /////////////////////////
%make LPF
radius = 10;
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
radius = 10;
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

%% Video Part
ds = 0.1;
%fn = 'BUbridge.mp4';
fn = 'Video.MOV';
frames = getVideoData(fn, ds);
refFrame = frames(:,:,1);
crosscorr = xcorr2(refFrame, frames(:,:,2));
figure
imagesc(crosscorr);

OneD_XC = crosscorr(:); % make cross correlation a 1D vector 
[~, maxix] = max(OneD_XC); %find max
[MaxRow, MaxCol] = ind2sub(size(crosscorr), maxix); 

Nframes = 339;
crosscorrmat = zeros((size(frames, 1)*2-1), (size(frames, 2)*2-1), Nframes);
Xdisp = zeros(1, Nframes);
Ydisp = zeros(1, Nframes);
for i=1:Nframes %length of frames
crosscorrmat(:,:,i) = xcorr2(refFrame, frames(:,:,i));
OneD_XC = crosscorrmat(:,:,i); % make cross correlation a 1D vector 
[~, maxx] = max(OneD_XC(:)); %find indicies of max of one frame cross corr
[Xdisp(i), Ydisp(i)] = ind2sub(size(crosscorrmat), maxx); % find values at the indicies
end

M = size(frames, 1);
N = size(frames, 2);
deltaM = -(Xdisp - M);
deltaN = -(Ydisp - N);

figure
plot(1:Nframes, deltaM, 'r') 
hold on;
plot(1:Nframes, deltaN, 'b')
xlabel('Frame #');
ylabel('Displacement (pixels)');
legend('X displacement', 'Y displacement');
hold off;

Movie = makeRegisteredMovie(frames,deltaM,deltaN);
writeRegisteredVideo('newmovie', Movie);

