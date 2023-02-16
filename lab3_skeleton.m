function lab3_skeleton()
clc; close all; 

data_path = fullfile(pwd);  %This line sets the path to the data as the directory containing the code

square = part3(data_path);  
[impulse, predicted] = parts4_5(data_path);
part6(square, impulse, predicted);





function out = part3(data_path)

%The names of your data files are specificied in this function.  Look at
%the structure 'fn' at the command line to see how file names are stored
fn = getFileNames();  

%Create variables that are the correct size containing zeros.  Not strictly
%necessary, but good practice.  There are as many elements in each matrix
%as there are file names for each experiment

gain.oneF = zeros(numel(fn.lpf_1uf), 1);  
gain.tenF = zeros(numel(fn.lpf_10uf), 1);
freq = zeros(numel(fn.nofilt), 1);


r = 3;  %Number of rows of subplots
c = 5;  %Number of columns of subplots
figure(1); 
for i = 1:numel(fn.nofilt)

    %Load the data for the first frequency into three structures
    nofilt = load(fullfile(data_path, fn.nofilt{i}));
    lpf1 = load(fullfile(data_path, fn.lpf_1uf{i}));
    lpf10 = load(fullfile(data_path, fn.lpf_10uf{i}));
    
    %You will need to add code here to compute the frequency of the
    %unfiltered waveform and gain of the two low-pass filters
    threshold = 2.5;
    ix = nofilt.ydata(1:end-1)<threshold & nofilt.ydata(2:end)>threshold;
    thisFrequency = 1./mean(diff(nofilt.xdata(ix)));
    
    gain.oneF(i) = (max(lpf1.ydata)-min(lpf1.ydata))/(max(nofilt.ydata)-min(nofilt.ydata));
    gain.tenF(i) = (max(lpf10.ydata)-min(lpf10.ydata))/(max(nofilt.ydata)-min(nofilt.ydata));
    freq(i) = thisFrequency;

    %Plot the unfiltered data.  You will need to make the plots look nicer
    subplot(r,c,i);  
    plot(nofilt.xdata, nofilt.ydata, 'LineWidth', 1.5);
    axis([-1 1 0 6])
    title(['Freq=' num2str(freq(i))]); 
    ylabel('Input Voltage (V)')
    set(gca,'FontSize',10)
    
    %You will need to add code here to plot the two sets of low-pass filtered
    %data
    %uF1 data
    subplot(r,c,i+5);  
    plot(lpf1.xdata, lpf1.ydata, 'LineWidth', 1.5);
    axis([-1 1 0 6])
    title(['Gain=' num2str(gain.oneF(i))]); 
    ylabel('Input Voltage (V)')
    set(gca,'FontSize',10)
    
    %uF10 data
    subplot(r,c,i+10);  
    plot(lpf10.xdata, lpf10.ydata, 'LineWidth', 1.5);
    axis([-1 1 0 6])
    title(['Gain=' num2str(gain.tenF(i))]); 
    ylabel('Input Voltage (V)')
    set(gca,'FontSize',10) 
    

end

figure(2); 
r = 1;
c = 5;
xl = [0.1 250];
yl = [1e-2 1e4];
for i = 1:numel(fn.nofilt)

    %Load the unfiltered data
    nofilt = load(fullfile(data_path, fn.nofilt{i}));
    
    %Compute an FFT of the unfiltered data
    [nofilt.Y,nofilt.f] = easyFFT(nofilt.ydata, numel(nofilt.ydata), 2, 1./mean(diff(nofilt.xdata)));

    %Plot the magnitude of the FFT of the unfiltered data
    subplot(r,c,i);
    power = 1;
    semilogy(nofilt.f, abs(nofilt.Y)./power, 'LineWidth', 1.5); hold on;
    xlim(xl);
    ylim(yl);
    set(gca,'FontSize',15,'fontWeight','bold')
    
    %You will need to add analagous code for the filtered data
%Load the lowpass 1uF data
    lpf1 = load(fullfile(data_path, fn.lpf_1uf{i}));
    
    %Compute an FFT of the lowpass 1uF data
    [lpf1.Y,lpf1.f] = easyFFT(lpf1.ydata, numel(lpf1.ydata), 2, 1./mean(diff(lpf1.xdata)));

    %Plot the magnitude of the FFT of the lowpass 1uF data
    subplot(r,c,i);
    power = 1;
    semilogy(lpf1.f, abs(lpf1.Y)./power, 'LineWidth', 1.5); hold on;
    xlim(xl);
    ylim(yl);
    xlabel('Frequency(Hz)');
    ylabel('FFT Amplitude');
    title(['Freq=' num2str(freq(i))]);
    set(gca,'FontSize',15,'fontWeight','bold')

    %Load the lowpass 10uF data
    lpf10 = load(fullfile(data_path, fn.lpf_10uf{i}));
    
    %Compute an FFT of the lowpass 10uF data
    [lpf10.Y,lpf10.f] = easyFFT(lpf10.ydata, numel(lpf10.ydata), 2, 1./mean(diff(lpf10.xdata)));

    %Plot the magnitude of the FFT of the lowpass 10uF data
    subplot(r,c,i);
    power = 1;
    semilogy(lpf10.f, abs(lpf10.Y)./power, 'LineWidth', 1.5); hold on;
    xlim(xl);
    ylim(yl);
    set(gca,'FontSize',10,'fontWeight','bold')
    
    %Add a legend
    legend('No filter', 'LPF 1 \muF', 'LPF 10 \muF', 'Location', 'northeast');
end

%Plot the gain of the filters at each frequency
figure(3);
semilogx(freq, gain.oneF, 'r.-', 'MarkerSize', 30);
grid on;
hold on;
semilogx(freq, gain.tenF, 'b.-', 'MarkerSize', 30);
legend('1 \muF', '10 \muF', Location='best')
title('Gain vs Frequency (Hz)')
xlabel('Square Wave Frequency')
ylabel('Gain')
set(gca,'FontSize',15,'fontWeight','bold') 



%Create the output structure
out.freq = freq;
out.gain = gain;



function [impulse, predicted] = parts4_5(data_path)

%The names of your data files are specificied in this function.  Look at
%the structure 'fn' at the command line to see how file names are stored
fn = getFileNames();

%Load the impulse response for the low pass filter with 1 uF capacitor
kern1 = load(fullfile(data_path, fn.kern_1uf));

%Plot the data
figure(4); hold on;
plot(kern1.xdata, kern1.ydata, 'r');
xlabel('Time (s)')
ylabel('Voltage (V)')
legend('1 \muF', '10 \muF', Location='best');
set(gca,'FontSize',15,'fontWeight','bold') 

%Crop the waveform starting at the first peak
[~, locs] = findpeaks(kern1.ydata, 'MinPeakHeight',0.1);
startix = locs(1);
endix = startix+1000-1;
kern1.wave = kern1.ydata(startix:endix);

%Plot the cropped impulse response
figure(5); hold on;
tm = kern1.xdata(1:numel(kern1.wave)) - kern1.xdata(1);
plot(tm, kern1.wave, 'r', 'LineWidth', 1.5);

%Add zeros to the end of the impulse response to make it 8 seconds long
kern1.wave(endix+1:4000) = 0;

%Plot the new, longer impulse response
figure(6); hold on;
tm = kern1.xdata(1:numel(kern1.wave)) - kern1.xdata(1);
plot(tm, kern1.wave, 'r', 'LineWidth', 1.5);

%Calculate the FFT of the impulse response
[kern1.fft,kern1.freq] = easyFFT(kern1.wave, numel(kern1.ydata), 2, 1./mean(diff(kern1.xdata)));

%Plot the normalized FFT magnitude of the impulse response.
figure(7); 
semilogx(kern1.freq, abs(kern1.fft)./max(abs(kern1.fft)), 'r-', 'LineWidth', 1); hold on;
set(gca,'FontSize',15,'fontWeight','bold')


%You will have to add code for the impulse response of the filter with a 10
%uF capacitor
%%
%Load the impulse response for the low pass filter with 1 uF capacitor
kern10 = load(fullfile(data_path, fn.kern_10uf));

%Plot the data
figure(4); hold on;
plot(kern10.xdata, kern10.ydata, 'b');
legend('1 \muF','10 \muF', Location='best');
set(gca,'FontSize',15,'fontWeight','bold') 

%Crop the waveform starting at the first peak
[~, locs] = findpeaks(kern10.ydata, 'MinPeakHeight',0.1);
startix = locs(1);
endix = startix+1000-1;
kern10.wave = kern10.ydata(startix:endix);

%Plot the cropped impulse response
figure(5); hold on;
tm = kern10.xdata(1:numel(kern10.wave)) - kern10.xdata(1);
plot(tm, kern10.wave, 'b', 'LineWidth', 1.5);
legend('1 \muF','10 \muF');
title('Impulse Response: Cropped Wave')
ylabel('Voltage(V)');
xlabel('Time(s)');
set(gca,'FontSize',15,'fontWeight','bold') 

%Add zeros to the end of the impulse response to make it 8 seconds long
kern10.wave(endix+1:4000) = 0;

%Plot the new, longer impulse response
figure(6); hold on;
tm = kern10.xdata(1:numel(kern10.wave)) - kern10.xdata(1);
plot(tm, kern10.wave, 'b', 'LineWidth', 1.5);
legend('1 \muF','10 \muF', Location='best');
title('Impulse Response: Elongated Wave')
ylabel('Voltage(V)');
xlabel('Time(s)');
set(gca,'FontSize',15,'fontWeight','bold') 

%Calculate the FFT of the impulse response
[kern10.fft,kern10.freq] = easyFFT(kern10.wave, numel(kern10.ydata), 2, 1./mean(diff(kern10.xdata)));

%Plot the normalized FFT magnitude of the impulse response.
figure(7); 
semilogx(kern10.freq, abs(kern10.fft)./max(abs(kern10.fft)), 'b-', 'LineWidth', 1); hold on;
xlabel('Square Wave frequency (Hz)')
ylabel('Gain')
title('Gain vs Frequncy')
legend('1 \muF','10 \muF', Location='best');
set(gca,'FontSize',15,'fontWeight','bold') 
%%
impulse.freq = kern1.freq;  %save the frequencies at which the FFT is calculated  to a structure
impulse.gain.oneF = kern1.fft;  %save the FFT of the impulse response
impulse.gain.tenF = 0;  %Fill this in appropriately

%Save analagous variables for the predicted frequency response of the
%low-pass filter.  Ensure that it contains the correct resistor and
%capacitor values
R = 10000;
C_1 = 0.000001;
C_10 = 0.00001;

predicted.freq = kern1.freq;
predicted.gain.oneF = 1./sqrt(1+(2*pi*kern1.freq*R.*C_1).^2);
predicted.gain.tenF = 1./sqrt(1+(2*pi*kern1.freq*R.*C_10).^2);







function part6(square, impulse, predicted)

%%
figure(8); 
%Plot the frequency response of the low-pass filter with 1uF capacitor
%using each of the three methods
semilogx(impulse.freq, abs(impulse.gain.oneF)./max(abs(impulse.gain.oneF)), 'r'); hold on;
semilogx(square.freq, abs(square.gain.oneF), 'r.-', 'MarkerSize', 30);
semilogx(predicted.freq, abs(predicted.gain.oneF)./max(abs(predicted.gain.oneF)), 'r--', 'LineWidth', 1.5);

% 10uF
semilogx(impulse.freq, abs(impulse.gain.tenF)./max(abs(impulse.gain.tenF)), 'b');
semilogx(square.freq, abs(square.gain.tenF), 'b.-', 'MarkerSize', 30); 
semilogx(predicted.freq, abs(predicted.gain.tenF)./max(abs(predicted.gain.tenF)), 'b--', 'LineWidth', 1.5);
grid on;
xlabel('Frequency(Hz)');
ylabel('Gain')
title('Gain vs. Frequency');
legend('1 \muF', '', '', '10 \muF');
set(gca,'FontSize',15,'fontWeight','bold') 
hold off;



%%
function fn = getFileNames()
%Create variables containing the file names for each data set
fn.nofilt{1} = 'nofilt_freq1.mat';
fn.nofilt{2} = 'nofilt_freq2.mat';
fn.nofilt{3} = 'nofilt_freq3.mat';
fn.nofilt{4} = 'nofilt_freq4.mat';
fn.nofilt{5} = 'nofilt_freq5.mat';

fn.lpf_1uf{1} = 'lpf_1uF_freq1.mat';
fn.lpf_1uf{2} = 'lpf_1uF_freq2.mat';
fn.lpf_1uf{3} = 'lpf_1uF_freq3.mat';
fn.lpf_1uf{4} = 'lpf_1uF_freq4.mat';
fn.lpf_1uf{5} = 'lpf_1uF_freq5.mat';

fn.lpf_10uf{1} = 'lpf_10uF_freq1.mat';
fn.lpf_10uf{2} = 'lpf_10uF_freq2.mat';
fn.lpf_10uf{3} = 'lpf_10uF_freq3.mat';
fn.lpf_10uf{4} = 'lpf_10uF_freq4.mat';
fn.lpf_10uf{5} = 'lpf_10uF_freq5.mat';

fn.kern_1uf = 'impulse_1uf.mat';
fn.kern_10uf = 'impulse_10uf.mat';




function [Y,f] = easyFFT( X, n, dim, samplerate )
%easyFFT Easy-to-use Fast Fourier Transform
%  Conveniently returns the frequency vector along with the FFT. 
%
%  Input Arguments
%   - SIG        : Input array, signal in time domain, equidistant sampling 
%   - n          : Transform length 
%   - dim        : Dimension to operate along
%   - samplerate : Samplig rate of X in Hz 
%
%  Output Arguments
%   - Y          : Frequency domain representation (fftshift'ed format) 
%   - f          : Frequency vector along dim
%
%  Usage Example
%   >> [Y,f] = easyFFT( X, 2^20, 1, 1/dt );
%
% arguments  
%   X                double {mustBeNumeric,mustBeReal}        % Input array
%   n          (1,1) double {mustBeInteger,mustBeNonnegative} % Transformation length TODO accept also []  
%   dim        (1,1) double {mustBeInteger,mustBePositive   } % Dimension to operate along
%   samplerate (1,1) double {              mustBePositive   } % sampling rate of X 
% end
% Perform FFT
X = X-mean(X);
% X = X./var(X);
Y = fft(X,n,dim); 
% Determine frequency vector
nSamples = size(Y, dim); % number of samples
df = samplerate / nSamples;
f = 0 : df : df*(nSamples-1);

f = f(1:floor(end/2));
Y = Y(1:floor(end/2));

