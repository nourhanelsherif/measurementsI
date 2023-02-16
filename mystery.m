figure;
load("mystery.mat");
plot(xdata,ydata, 'b', 'Linewidth', 2);
xlabel("Time (s)");
ylabel("Signal");
meanVal = mean(ydata)
title(sprintf("Mean = %.1f", meanVal));

sfrq=500; %depends on the given hz
[peaks, locs] = findpeaks(ydata, 'MinPeakHeight', 8);
period = diff(locs/sfrq); % convert indices to t vals
frq = 1/mean(period)