phase_matching_matrix1=interp1(dk_draw,phase_matching1,delta_k_matrix);
figure;
subplot 121;
imagesc(2*pi*c./(w_s_array)*1e6, 2*pi*c./(w_i_array)*1e6, phase_matching_matrix.*pump);
xlabel('\lambda Signal \mum');
ylabel('\lambda  Idler \mum');
set(gca,'YDir','normal'); 
axis equal; title('modify poling JSA '); %axis([-18 18 -18 18]);
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

subplot 122;
imagesc(2*pi*c./(w_s_array)*1e6, 2*pi*c./(w_i_array)*1e6, phase_matching_matrix1.*pump);
xlabel('\lambda Signal \mum');
ylabel('\lambda  Idler \mum');
set(gca,'YDir','normal'); 
axis equal; title('regular JSA'); %axis([-18 18 -18 18]);
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

text1=['Pump' num2str(lambda_p*1e9) 'nm FWHM=1nm to' ...
    num2str(lambda_s*1e9) 'nm poling '...
    num2str(round(-2*pi/poling_freq*1e7)/10) ...
    ' micron crystal size ' num2str(round(MaxZ*10000)/10) 'mm'];
annotation('textbox', [0 0.9 1 0.1], ...
    'String',text1, ... 
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center');