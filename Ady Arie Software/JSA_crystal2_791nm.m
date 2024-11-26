%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create tailored JSA according to spectral holograms

% Inbar Hurvitz, December 2021
% This code calculates JSA and output of Sivan's Crystal (#2, 15.9 microns)
% this is for a pump of 791.15nm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear all; 
%close all; %pack;

%general definitions
c             = 2.99792458e8;%in meter/sec 
deff          = 3.7*1e-12;    % d24 (y->y+z) in pico-meter/Volt.[KTP] (J. D. Bierlein and H. Vanherzeele, ?Potassium titanyl phosphate: properties and new applications,? J. Opt. Soc. Am. B, vol. 6, no. 4, pp. 622?633, Apr. 1989.)
eps0          = 8.854187817e-12; % the vacuum permittivity, in Farad/meter.
I             = @(A,n) 2.*n.*eps0.*c.*abs(A).^2;  
h_bar         = 1.054571800e-34; % Units are m^2 kg / s, taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck
Fourier       = @(A) (fftshift(fft2(ifftshift(A))))/sqrt(numel(A)); %Fourier and normalize
v_g           = @(fun,lam,T) -(fun((lam+0.1e-9)*1e6,T)*(lam-0.1e-9)- fun((lam-0.1e-9)*1e6,T)*(lam+0.1e-9))/c/0.2e-9;
E0            = @(P,n,W0) sqrt(P./(n*c*eps0*pi*W0^2)) ;

%for calculating the refractive index, use Fradkin for z and Wong for y
n_y_by_wong   = @(lambda) sqrt(2.09930 + 0.922683./(1 - 0.0467695*lambda.^-2) - 0.0138404*lambda.^2);
%pump_crystal  = @(lambda,T) n_y_by_wong(lambda)+dn_dty1(lambda,T);
%idler_crystal = @(lambda,T) n_y_by_wong(lambda)+dn_dty1(lambda,T);
%signal_crystal= @(lambda,T) nz_KTP_Arie(lambda,T);
pump_crystal  = @(lambda,T) n_y_by_wong(lambda)+dn_dty(lambda,T);
idler_crystal = @(lambda,T) n_y_by_wong(lambda)+dn_dty(lambda,T);
signal_crystal= @(lambda,T) n_ktp_5gT(lambda,T);

%structure arrays
dx            = 1e-6; 
dy            = 1e-6;
MaxX          = 400e-6; x = -MaxX:dx:MaxX-dx;
MaxY          = 400e-6; y = -MaxY:dy:MaxY-dy;
Power2D       = @(A,n) sum(sum(I(A,n)))*dx*dy;

T               = 25; %Celsius

%pump
lambda_p        = 791.15e-9; %titanium Saphhir laser main wavelength
% lambda_p    = 780e-9;
w_p             = 2*pi*c/lambda_p;
n_p             = pump_crystal(lambda_p*1e6,T);
n_p0            = pump_crystal(lambda_p*1e6,25);
k_p             = 2*pi*n_p/lambda_p;
k_p0            = 2*pi*n_p0/lambda_p;
v_g_p           = v_g(pump_crystal,lambda_p,T);

%signal
lambda_s        = lambda_p*2;
w_s             = 2*pi*c/lambda_s;
n_s             = signal_crystal(lambda_s*1e6,T);
n_s0            = signal_crystal(lambda_s*1e6,25);
k_s             = 2*pi*n_s/lambda_s;
k_s0            = 2*pi*n_s0/lambda_s;
v_g_s           = v_g(signal_crystal,lambda_s,T);

%idler
lambda_i        = lambda_p*2;
w_i             = 2*pi*c/lambda_i;
n_i             = idler_crystal(lambda_i*1e6,T);
n_i0            = idler_crystal(lambda_i*1e6,25);
k_i             = 2*pi*n_i/lambda_i;
k_i0            = 2*pi*n_i0/lambda_i;
v_g_i           = v_g(idler_crystal,lambda_i,T);

%width of pump distribution
FWHM               = 1.5e-9;         %Full width half maximum in the Dosseva paper
% FWHM               = 0.8e-9;
delta_lambda       = FWHM/2/sqrt(2*log(2)); %move to sigma by relationship to FWHM of a Gaussian
lambda_min         = lambda_p-delta_lambda;
lambda_max         = lambda_p+delta_lambda;
width_in_freq_pump = 2*pi*c*delta_lambda/(lambda_min*lambda_max);%convert to 1/s

%calculate ideal length of crystal
gamma              = 0.193; %factor for matching a sinc to a gaussian with the same FWHM, see Branczyk
L                  = sqrt(8/gamma/width_in_freq_pump^2/(v_g_i-v_g_s)^2); %condition for sinc=gaussian
factor             = 1.4;
MaxZ               = L*factor;   
%dz                 = 2e-6;
dz                 = 5e-6; %Sivan used 2e-6.
z                  = -MaxZ/2:dz:MaxZ/2-dz;
[X,Z]              = meshgrid(x,z);
N                  = length(z);

%build arrays
N_new = 101;
num = 3;
w_s_array       = linspace(w_s-num*width_in_freq_pump,w_s+num*width_in_freq_pump,N_new);
lambda_s_array  = 2*pi*c./w_s_array;
k_s_array       = 2*pi*signal_crystal(lambda_s_array*1e6,T)./lambda_s_array;

w_i_array       = linspace(w_i-num*width_in_freq_pump,w_i+num*width_in_freq_pump,N_new);
lambda_i_array  = 2*pi*c./w_i_array;
k_i_array       = 2*pi*idler_crystal(lambda_i_array*1e6,T)./lambda_i_array;


[W_i, W_s]      = meshgrid(w_i_array,w_s_array);
W_p             = W_s+W_i;
pump            = exp(-(W_p-w_p).^2/2/(width_in_freq_pump^2));

k_i_matrix      = 2*pi*idler_crystal(2*pi*c./W_i*1e6,T)./(2*pi*c./W_i);
k_s_matrix      = 2*pi*signal_crystal(2*pi*c./W_s*1e6,T)./(2*pi*c./W_s);
k_p_matrix      = 2*pi*pump_crystal(2*pi*c./W_p*1e6,T)./(2*pi*c./W_p);
delta_k_matrix  = k_p_matrix - k_s_matrix - k_i_matrix;

% plot the pump distribution - lambda
figure(); 
imagesc((lambda_i_array)*1e9, (lambda_s_array)*1e9, (pump));
xlabel('\lambda_i [nm]');
ylabel('\lambda_s [nm]');
set(gca,'YDir','normal'); colorbar; % colormap bone;
axis equal; title('\alpha(\lambda_s,\lambda_i), pump frequency distribution');
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

w_p_array      = w_s_array+w_i_array;
lambda_p_array = 2*pi*c./w_p_array;
k_p_array      = 2*pi*pump_crystal(lambda_p_array*1e6,T)./lambda_p_array;
% delta_k0       = k_p-k_s-k_i;
% delta_k_array  = k_p_array-k_i-k_s;
delta_k_array  = k_p-k_i_array-k_s_array;

%% crystal
%poling_freq   = delta_k0;
poling_freq = k_p0-k_s0-k_i0; %should be independent of temperature
delta_k0 = poling_freq;
%poling_freq = 2*pi/(46.2e-6);

% regular PP
phi           = 0;
q             = 1/2;

w_s_array_temp       = linspace(w_s-num*width_in_freq_pump,w_s+num*width_in_freq_pump,length(z));
lambda_s_array_temp  = 2*pi*c./w_s_array_temp;
k_s_array_temp       = 2*pi*signal_crystal(lambda_s_array_temp*1e6,T)./lambda_s_array_temp;

w_i_array_temp       = linspace(w_i-num*width_in_freq_pump,w_i+num*width_in_freq_pump,length(z));
lambda_i_array_temp  = 2*pi*c./w_i_array_temp;
k_i_array_temp       = 2*pi*idler_crystal(lambda_i_array_temp*1e6,T)./lambda_i_array_temp;

w_p_array_temp      = w_s_array_temp + w_i_array_temp ;
lambda_p_array_temp = 2*pi*c./w_p_array_temp;
k_p_array_temp      = 2*pi*pump_crystal(lambda_p_array_temp*1e6,T)./lambda_p_array_temp;

delta_k_array_temp  = k_p-k_i_array_temp-k_s_array_temp;


Gaussianwidth_in_k     = 2*pi*(pump_crystal(lambda_min*1e6,T)/lambda_min - pump_crystal(lambda_max*1e6,T)/lambda_max);
Gaussianwidth_in_k     = Gaussianwidth_in_k*2/factor; %why?
desired_beam_F         = exp(-((delta_k_array_temp-delta_k0).^2/2/(Gaussianwidth_in_k)^2));

%HG01 Sivan
% desired_beam_F= Hermite_gause(lambda_p, Gaussianwidth_in_k,  1,0, delta_k_array-delta_k0);

%HG01 Inbar
order=1;
Desired_beam_F=desired_beam_F.*Hermite((delta_k_array_temp-delta_k0)/Gaussianwidth_in_k,order);

% double Gaussian
% spacing = w_p/100;
% desired_beam_F= exp(-((w_p_array-w_p+spacing).^2/(width_in_freq/400)^2));
% desired_beam_F= desired_beam_F+ exp(-((w_p_array-w_p-spacing).^2/(width_in_freq/400)^2));

desired_beam  = (desired_beam_F);
desired_beam  = desired_beam/max(abs(desired_beam(:))); %normalize
phi           = angle(desired_beam);
q             = asin(abs(desired_beam))/pi;

PP            = sign(cos(poling_freq.*z+phi)-cos(pi*q));
PP_no         = sign(cos(poling_freq.*z));

figure(); 
plot(z*1e3,-q*2*pi/poling_freq*1e6); title('Duty cycle')
xlabel('z [mm]'); ylabel('domain width [\mum]')
set(findall(gcf,'-property','FontSize'),'FontSize',20);

figure(); 
H_poling=bar(z*1e3,0.5*(PP+1)); xlabel('z [mm]'); hold on;
set(H_poling,'FaceColor',[1,1,1]*0.7,'EdgeColor',[1,1,1]*0.7) %Fill bars in gray
plot(z*1e3,abs(desired_beam).^2,'LineWidth',2);
 set(findall(gcf,'-property','FontSize'),'FontSize',20);

%plot the fourier transform of the crystal, the phase matching function
times          = 10; %how much larger to make the arrays: new_size=old_size*2*times

pad_size       = 30000;
z_wide         = cat(1,z',linspace(max(z)+dz, max(z)+dz*pad_size,pad_size)');
z_wide         = cat(1,linspace(min(z)-dz*pad_size, min(z)-dz,pad_size)',z_wide);
PP_pad         = padarray(PP',pad_size,0,'both'); PP_pad=PP_pad';
PP_pda_no      = padarray(PP_no',pad_size,0,'both'); PP_pda_no=PP_pda_no';
dz_wide        = dz; 
L_wide         = max(z_wide)-min(z_wide);
dk_draw        = 2*pi*(z_wide-dz_wide/2)/(dz_wide*L_wide);%Fourier: k=2pi*z/(L*dz)
dk_draw_tag    = dk_draw-poling_freq;
phase_matching = abs(Fourier(PP_pad));
phase_matching1 = abs(Fourier(PP_pda_no));

dk_draw_tag_cut=dk_draw_tag(dk_draw_tag>min(delta_k_array)-poling_freq);
dk_draw_tag_cut=dk_draw_tag_cut(dk_draw_tag_cut< -min(delta_k_array)+poling_freq);
phase_matching_cut=phase_matching(dk_draw_tag>min(delta_k_array)-poling_freq);
phase_matching_cut=phase_matching_cut(dk_draw_tag_cut< -min(delta_k_array)+poling_freq);

phase_matching_cut1=phase_matching1(dk_draw_tag>min(delta_k_array)-poling_freq);
phase_matching_cut1=phase_matching_cut1(dk_draw_tag_cut< -min(delta_k_array)+poling_freq);

% plot figures
figure(); 
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

%%
phase_matching_fixed=interp1(dk_draw,phase_matching,delta_k_array);
phase_matching_matrix=interp1(dk_draw,phase_matching,delta_k_matrix);

figure();
imagesc((lambda_i_array)*1e9, (lambda_s_array)*1e9, phase_matching_matrix);
xlabel('\lambda_i [nm]');
ylabel('\lambda_s [nm]');
set(gca,'YDir','normal'); colorbar;
axis equal; title('\Phi(\lambda_s,\lambda_i), phase matching function');
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');

figure();
imagesc(lambda_i_array*1e9, lambda_s_array*1e9, abs(phase_matching_matrix.*pump));
xlabel('\lambda_i [nm]');
ylabel('\lambda_s [nm]');
colorbar;
set(gca,'YDir','normal'); % colormap bone;
title('JSA'); %axis([-18 18 -18 18]);
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% create DXF
% if 0
%     
% %calculate with higher resolution
% % make the resolution 0.025 microns!!!
% % FIXXX
% z_resolution         = linspace(min(z),max(z),length(z));
% dz_resolution        = z_resolution(2)-z_resolution(1);
% desired_beam_res     = interp1(z,desired_beam,z_resolution);
% phi_res              = angle(desired_beam_res);
% q_res                = asin(abs(desired_beam_res))/pi;
% 
% PP_resolution        = sign(cos(poling_freq.*z_resolution+phi_res)-cos(pi*q_res));
% 
% % PP_BW                = (PP_resolution+1)/2;%make binary for edge detection
% % PP_edges             = edge(PP_BW);
% 
% %opening the final DXF file
% filename  = 'JSA_DXF.dxf';
% [fid,err] = DXF_start(filename,1);
% 
% %general paramaters
% span_y                  = 1; %width of each design
% location_y              = 0; %the zero of the y location
% 
% %the y coordinates are always the same
% y                       = (location_y+[0,span_y,span_y,0,0]);
% 
% x_start                 = 0;
% flag                    = 0; %flag indicating that we have seen an edge
% 
% figure();
% bar(z_resolution-min(z_resolution),0.5*(PP_resolution+1)); hold on;
% 
% for n = 1:length(PP_resolution)    
%      
%      if PP_resolution(n) == -1 && ~flag %if there is no edge, and haven't found a first edge yet yet
%         x_start  = x_start + dz_resolution; %move the start
%         disp(1)
%      elseif PP_resolution(n) == 1 && ~flag %found a first edge
%          x_end = x_start + dz_resolution;
%          flag  = ~flag;
%      elseif PP_resolution(n) == 1 && flag %continue in segment
%          x_end = x_end + dz_resolution;
%      elseif PP_resolution(n) == -1 && flag %found a second edge         
%          %write domain to file
%          x     = [x_start,x_start,x_end,x_end,x_start];
%          plot(x,y,'--'); hold on;
%          disp('yey')
%          
%          x_start = x_end+ dz_resolution;
%          flag  = ~flag;
%          
%          DXF_poly(fid,x,y,5,7,1); 
% 
%      end
%     
%      
% end
% 
% DXF_end(fid);
% fclose('all');
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simualte output
% YZY in SHG

%light definitions
num_iter  = 500;
lambda1   = lambda_s; %central wavelength of pump,
l_width   = 100e-9; %the width of its spectrum
lambda1   = linspace(lambda1-l_width,lambda1+l_width,num_iter); 
lambda2   = lambda1/2; %SHG
W0_x      = 200e-6; 
W0_y      = 200e-6; 
W0        = sqrt(W0_x^2+W0_y^2);%for planewave these do not matter, just for calculating the power of a gaussian beam after...

n1_y             = pump_crystal(lambda1*1e6,T);
n1_z             = signal_crystal(lambda1*1e6,T);
n2_y             = pump_crystal(lambda2*1e6,T);

k1_y             = n1_y * 2 * pi ./ lambda1;
k1_z             = n1_z * 2 * pi ./ lambda1;
k2_y             = n2_y * 2 * pi ./ lambda2;

w1               =  2*pi*c ./ lambda1;
w2               =  2*pi*c ./ lambda2;

delta_k       = k1_y + k1_z - k2_y;

kappa1_y        = 2 * 1i * deff / c^2 * w1.^2 ./ k1_y;
kappa1_z        = 2 * 1i * deff / c^2 * w1.^2 ./ k1_z;
kappa2          = 2 * 1i * deff / c^2 * w2.^2 ./ k2_y;

input_pump_power = 1;
E_1_y        = E0(input_pump_power/2,n1_y,W0);
E_1_z        = E0(input_pump_power/2,n1_z,W0);
E_2_y        = 0; %SHG

%PP
E_1_y_PP        = E_1_y;
E_1_z_PP        = E_1_z;
E_2_y_PP        = E_2_y;

%propagate through crystasl
for n=1:length(z)
    z_tag=z(n);

    %Special crystal
    %Non-linear equations:
    dE1_y_dz     = kappa1_y.*PP(n).*conj(E_1_z).*E_2_y.*exp(-1i*delta_k*z_tag);
    dE1_z_dz     = kappa1_z.*(PP(n)).*conj(E_1_y).*E_2_y.*exp(-1i*delta_k*z_tag);
    dE2_y_dz     = kappa2.*(PP(n)).*E_1_y.*E_1_z.*exp(1i*delta_k*z_tag);

    %Add the non-linear part
    E_1_y        = E_1_y + dE1_y_dz*dz;
    E_1_z        = E_1_z + dE1_z_dz*dz;
    E_2_y        = E_2_y + dE2_y_dz*dz;
    
    %do the same for PP
    %Non-linear equations:
    %regular_poling  = sign(cos(abs(delta_k0)*z_tag));
    regular_poling  = sign(cos(abs(poling_freq)*z_tag));
    dE1_y_dz_PP     = kappa1_y.*regular_poling.*conj(E_1_z_PP).*E_2_y_PP.*exp(-1i*delta_k*z_tag);
    dE1_z_dz_PP     = kappa1_z.*regular_poling.*conj(E_1_y_PP).*E_2_y_PP.*exp(-1i*delta_k*z_tag);
    dE2_y_dz_PP     = kappa2.*regular_poling.*E_1_y_PP.*E_1_z_PP.*exp(1i*delta_k*z_tag);

    %Add the non-linear part
    E_1_y_PP        = E_1_y_PP + dE1_y_dz_PP*dz;
    E_1_z_PP        = E_1_z_PP + dE1_z_dz_PP*dz;
    E_2_y_PP        = E_2_y_PP + dE2_y_dz_PP*dz;
    
    % disp(['propagating, ', num2str(n/length(z)*100), '%']);
end

figure(); 
plot(lambda1*1e9,2.*n2_y.*eps0.*c.*abs(E_2_y).^2*W0);
hold on;
plot(lambda1*1e9,2.*n2_y.*eps0.*c.*abs(E_2_y_PP).^2*W0,'--k');
xlabel('Input wavelength [nm]');
ylabel('Output Power [W]')
legend('Output SH', 'Output SH for periodically poled')

% figure(); plot(lambda1*1e9,2.*n1_y.*eps0.*c.*abs(E_1_y).^2,lambda1*1e9,2.*n1_z.*eps0.*c.*abs(E_1_z).^2,'--');
% xlabel('Input wavelength [nm]');
%% Schmidt decomposition
[U S V]=rsvd(phase_matching_matrix.*pump, length(pump));
figure(10); plot(diag(S(1:20,1:20))/sum(diag(S)),'*'); title('Schmidt coefficients'); grid on;


b = diag(S(1:20,1:20))/sum(diag(S));
purity = sum(b.^4)
entanglement = sum(-b.^2 .* log2(b.^2));
