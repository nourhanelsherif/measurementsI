%% Part 2
A = load("Rainbow.mat");

subplot(2,2,1);
plot(A.xdata, A.ydata(1,:), 'r', LineWidth= 1);
hold on;
plot(A.xdata, A.ydata(2,:), 'g', LineWidth= 1);
plot(A.xdata, A.ydata(3,:), 'b', LineWidth= 1);
hold off;
axis([-8 7 0 1])
xlabel('Sample Number');
ylabel('Color Value');
title('Rainbow.mat');
set(gca,'FontSize',12,'fontWeight','bold','YDir', 'normal');

%Assuming startix and endix represent the starting and ending %indices found in step 3
ix = 1:190;
npts = numel(ix);
dat = zeros(npts, npts, 3);
dat(:, :, 1) = repmat(A.ydata(1,ix), npts, 1);
dat(:, :, 2) = repmat(A.ydata(2,ix), npts, 1);
dat(:, :, 3) = repmat(A.ydata(3,ix), npts, 1);

subplot(2,2,3);
imagesc(dat)
title('rainbrow.mat');
xlabel('Sample Number');
ylabel('Arbitrary units');
set(gca,'FontSize',12,'fontWeight','bold');

%RedBlueWhite
B = load("RedBlueWhite.mat");
subplot(2,2,2);
plot(B.xdata, B.ydata(1,:), 'r');
hold on;
plot(B.xdata, B.ydata(2,:), 'g');
plot(B.xdata, B.ydata(3,:), 'b');
hold off;
axis([-8 5 0 1])
xlabel('Sample Number');
ylabel('Color Value');
title('redblue.mat');
set(gca,'FontSize',12,'fontWeight','bold','YDir', 'normal');

%Assuming startix and endix represent the starting and ending %indices found in step 3
ix_B = 20:200;
npts_B = numel(ix_B);
datB = zeros(npts_B, npts_B, 3);
datB(:, :, 1) = repmat(B.ydata(1,ix_B), npts_B, 1);
datB(:, :, 2) = repmat(B.ydata(2,ix_B), npts_B, 1);
datB(:, :, 3) = repmat(B.ydata(3,ix_B), npts_B, 1);

subplot(2,2,4);
imagesc(datB)
title('redblue.mat');
xlabel('Sample Number');
ylabel('Arbitrary units');
set(gca,'FontSize',12,'fontWeight','bold');

%% Part 4
C = imread("GridSmall.jpeg");
D = imread("GridBig.jpeg");
E = imread("StripesVertical.jpeg");
F = imread("StripesDiag.jpeg");


im = normalizeImage(C);
Cfig1=subplot(2,4,1);
imagesc(im);
colormap(Cfig1, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')

Cfig2 =subplot(2,4,5);
FFTC = get2DFFT(im);
lim = linspace(0, 0.5, length(FFTC));
imagesc(lim, lim, FFTC);
colormap(Cfig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')


imD = normalizeImage(D);
Dfig1=subplot(2,4,2);
imagesc(imD);
colormap(Dfig1, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')

Dfig2 =subplot(2,4,6);
FFTD = get2DFFT(imD);
limD = linspace(0, 0.5, length(FFTD));
imagesc(limD, limD, FFTD);
colormap(Dfig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')

% E
imE = normalizeImage(E);
Efig1=subplot(2,4,3);
imagesc(imE);
colormap(Efig1, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')

Efig2 =subplot(2,4,7);
FFTE = get2DFFT(imE);
limE = linspace(0, 0.5, length(FFTE));
imagesc(limE, limE, FFTE);
colormap(Efig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')

% F
imF = normalizeImage(F);
Ffig1=subplot(2,4,4);
imagesc(imF);
colormap(Ffig1, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')

Ffig2 =subplot(2,4,8);
FFTF = get2DFFT(imF);
limF = linspace(0, 0.5, length(FFTF));
imagesc(limF, limF, FFTF);
colormap(Ffig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')

%%
sz = 151;
x = 1:sz;
y = 0.5+0.5*sin(x/2);

imV = repmat(y, sz, 1);
Vfig=subplot(2,4,1);    %vertical
imagesc(imV);
colormap(Vfig, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')

%FFT
Vfig2 =subplot(2,4,5);
FFTV = get2DFFT(imV);
lim = linspace(0, 0.5, length(FFTV));
imagesc(lim, lim, FFTV);
colormap(Vfig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')

imH = imrotate(im1, 90, 'bilinear', 'crop');
Hfig=subplot(2,4,2);    %Horizontal
imagesc(imH);
colormap(Hfig, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')


%FFT 
Hfig2 =subplot(2,4,6);
FFTH = get2DFFT(imH);
lim = linspace(0, 0.5, length(FFTH));
imagesc(lim, lim, FFTH);
colormap(Hfig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')

imDG = zeros(sz,sz);
imDG(1, 1) = 1;
DGfig=subplot(2,4,3);    %Diagonal
imagesc(imDG);
colormap(DGfig, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')


%FFT
DGfig2 =subplot(2,4,7);
FFTDG = get2DFFT(imDG);
lim = linspace(0, 0.5, length(FFTDG));
imagesc(lim, lim, FFTDG);
colormap(DGfig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')

[X,Y] = meshgrid(1:sz,1:sz);
X = X - mean(X(:));
Y = Y - mean(Y(:));
imCirc = 0.5+0.5.*sin(sqrt((X.^2+Y.^2))./2);

Circfig=subplot(2,4,4);    %circle
imagesc(imCirc);
colormap(Circfig, gray);
set(gca, 'YDir', 'normal');
title('Space domain');
xlabel('X position (pixels)')
ylabel('Y position (pixels)')


%FFT
Circfig2 =subplot(2,4,8);
FFTCirc = get2DFFT(imCirc);
lim = linspace(0, 0.5, length(FFTCirc));
imagesc(lim, lim, FFTCirc);
colormap(Circfig2, "jet")
set(gca, 'YDir', 'normal');
colorbar;
caxis([0, 1500]);
title('Frequency domain')
xlabel('X frequency (cycles/pix)')
ylabel('Y frequency (cycle/pix)')
