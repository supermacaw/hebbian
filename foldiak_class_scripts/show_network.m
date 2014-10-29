% show_network.m - display network parameters
%

function show_network(theta,Y_ave,W,Cyy_ave,Q)

figure(1)

M=size(Q,1);

subplot(321)
bar(theta);
set(gca,'XLim',[0 M+1])
title('theta')

subplot(322)
bar(Y_ave)
set(gca,'XLim',[0 M+1])
title('Y\_ave')

subplot(323)
imagesc(-W), colorbar, axis image
title('-W')

subplot(324)
C=Cyy_ave-Y_ave*Y_ave';
imagesc(C,[0 max(C(:))]), colorbar, axis image
title('Cyy\_ave')

subplot(325)
showrfs(Q)
title('Q')

drawnow
