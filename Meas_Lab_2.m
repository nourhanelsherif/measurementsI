%read in
A = load('0.01uF.mat');
B = load('0.1uF.mat');
C = load("1uF.mat");
D = load("10uF.mat");
E = load("100uF.mat");

%plot each wave (side by side in a subplot)
figure
subplot(1,5,1);
plot(A.xdata,A.ydata, 'r');
title('C = 0.01 uF');
xlabel('Time (s)');
ylabel('Voltage (V)');
subplot(1,5,2);
plot(B.xdata,B.ydata, 'b');
title('C = 0.1 uF');
xlabel('Time (s)');
ylabel('Voltage (V)');
subplot(1,5,3);
plot(C.xdata,C.ydata, 'g');
title('C = 1 uF');
xlabel('Time (s)');
ylabel('Voltage (V)');
subplot(1,5,4);
plot(D.xdata,D.ydata, 'k');
title('C = 10 uF');
xlabel('Time (s)');
ylabel('Voltage (V)');
subplot(1,5,5);
plot(E.xdata,E.ydata, 'm');
title('C = 100 uF');
xlabel('Time (s)');
ylabel('Voltage (V)');

%part 3 

%peak to peak
VoA= peak2peak(A.ydata);
VoB= peak2peak(B.ydata);
VoC= peak2peak(C.ydata);
VoD= peak2peak(D.ydata);
VoE= peak2peak(E.ydata);

%initialize vars 
F= 1/0.05; % 20 hz: frequency based on square wave
W=2*pi*F; %omega
R=10000; %resistor
%capacitors
CA=0.01*(10^(-6));
CB=0.1*(10^(-6));
CC=1*(10^(-6));
CD=10*(10^(-6));
CE=100*(10^(-6));

%gain equations
ExpGainA = 1/(sqrt(1+(W*R*CA)^2));
ExpGainB = 1/(sqrt(1+(W*R*CB)^2));
ExpGainC = 1/(sqrt(1+(W*R*CC)^2));
ExpGainD = 1/(sqrt(1+(W*R*CD)^2));
ExpGainE = 1/(sqrt(1+(W*R*CE)^2));

Peaks = [VoA VoB VoC VoD VoE]/5;
Capacitors = [CA CB CC CD CE];
Gains = [ExpGainA ExpGainB ExpGainC ExpGainD ExpGainE];

%plotting
figure
semilogx(Capacitors, Peaks, 'r', 'LineWidth', 2);
title('Gain vs Capacitor Value for 20 Hz square wave');
xlabel('Capacitor Values (uF)');
ylabel('Gain');
grid on
hold on
semilogx(Capacitors, Gains, 'ko');
legend('Measured Gain','Predicted Gain')
hold off


%%
load steps.mat;
plot(xdata,ydata);
Fc = 2.5; %Cutoff frequency
order = 2; %filter order
Fs = 1./mean(diff(xdata)); %Sampling frequency
Fn = Fc./(0.5*Fs); %Normalized cutoff frequency
[b, a] = butter(order, Fn, 'low');

[H, W] = freqz(b, a, [], Fs);
Habs= abs(H);
figure
plot(W, Habs);

filtdata = filtfilt(b, a, ydata);
figure
plot(xdata, ydata, 'k', LineWidth=1);
hold on
[peaks, locs] = findpeaks(filtdata, 'MinPeakDistance',7);
plot(xdata, filtdata, 'r', LineWidth=2);
xlabel('Time (sec)', FontWeight='bold');
ylabel('Acceleration (Gs)', FontWeight='bold');
legend('Raw Data', 'Filtered Data', 'Location','best');
title('Accelerometer: Raw and Filtered Data');
hold off
figure
plot(xdata, filtdata, 'r', LineWidth=2);
hold on
plot(xdata(locs), filtdata(locs), 'k.', 'MarkerSize', 20);
xlabel('Time (sec)', FontWeight='bold');
ylabel('Acceleration (Gs)', FontWeight='bold');
legend('Peak Locations', 'Filtered Data', 'Location','best');
title('26 Steps/1.625 steps/sec');






