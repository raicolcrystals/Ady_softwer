phase_matching_matrix1=interp1(dk_draw,phase_matching1,delta_k_matrix);
figure;
subplot 221;
imagesc(2*pi*c./(w_s_array)*1e6, 2*pi*c./(w_i_array)*1e6, phase_matching_matrix.*pump);
xlabel('\lambda Signal \mum');
ylabel('\lambda  Idler \mum');
set(gca,'YDir','normal'); 
axis equal; title('JSA'); %axis([-18 18 -18 18]);
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

subplot 222;
imagesc(2*pi*c./(w_s_array)*1e6, 2*pi*c./(w_i_array)*1e6, phase_matching_matrix1.*pump);
xlabel('\lambda Signal \mum');
ylabel('\lambda  Idler \mum');
set(gca,'YDir','normal'); 
axis equal; title('JSA'); %axis([-18 18 -18 18]);
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

subplot 223;
plot(dk_draw_tag_cut*1e-6, phase_matching_cut/max(phase_matching(:)), 'LineWidth',3);
hold on; plot(dk_draw_tag_cut*1e-6, phase_matching_cut1/max(phase_matching_cut1(:)))
% xlim([-2*pi*abs((poling_freq))*3,2*pi*abs((poling_freq+abs(delta_k0)))*3]*1e-6);
xlabel('\Deltak'' [rad/\mum]'); 
ylabel('|\Phi(\Deltak'')| [A.U]');
title('Fourier Transform of the crystal');
% xlim([-0.02 0.02])
axis([-0.015 0.015 0 1.1])
set(findall(gcf,'-property','FontSize'),'FontSize',20);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

subplot 224;
plot(lambda1*1e9,2.*n2_y.*eps0.*c.*abs(E_2_y).^2*W0);
hold on;
plot(lambda1*1e9,2.*n2_y.*eps0.*c.*abs(E_2_y_PP).^2*W0,'--k');
xlim([min(lambda1*1e9)+1.5*std(lambda1*1e9) max(lambda1*1e9)-1.5*std(lambda1*1e9)])
xlabel('Input wavelength [nm]');
ylabel('Output Power [W]')
legend('Output SH', 'Output SH for periodically poled')