load respiration.mat;
figure
plot(xdata, ydata, 'r','LineWidth', 1.5)
ylabel('Temperature (deg C)')
xlabel('Time (sec)')
title('Respiration Data')
xticks([-20 -10 0 10 20])
yticks([27.5 28.5 29.5 30.5 31.5 32.5])

disp(['Average = ' num2str(mean(ydata))]);
disp(['Maximum = ' num2str(max(ydata))]);
disp(['Minimum = ' num2str(min(ydata))]);

[peaks, locs] = findpeaks(ydata, 'MinPeakDistance',50);
figure; hold on;
plot(xdata, ydata);
plot(xdata(locs), ydata(locs), 'k.', 'MarkerSize', 20)

peak_times = xdata(locs);
periods = diff(peak_times);
disp(['Average Period = ' num2str(mean(periods)) ' seconds']);
disp(['Average Frequency = ' num2str(1/mean(periods)) ' Hz']);
title('Respiration Data with Local Maximas')
ylabel('Temperature (deg C)')
xlabel('Time (sec)')

figure;
bins = 2.6:0.4:6;
hist(periods, bins);
title(['Respiration rate = ' num2str(1/mean(periods)) ' Hz']);
xlabel('respiration periods')
ylabel('num of breaths')