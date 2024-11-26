%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find point where GVM = 0 for KTP type 2
%
% Sivan Trajtenberg-Mills, Dec. 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; pack;

%general definitions
c             = 2.99792458e8;%in meter/sec 
deff          = 7.6*1e-12;    % d24 (y->y+z) in pico-meter/Volt.[KTP]
eps0          = 8.854187817e-12; % the vacuum permittivity, in Farad/meter.
I             = @(A,n) 2.*n.*eps0.*c.*abs(A).^2;  
h_bar         = 1.054571800e-34; % Units are m^2 kg / s, taken from http://physics.nist.gov/cgi-bin/cuu/Value?hbar|search_for=planck

T             = 25; %Celsius

N               = 2000;

%for calculating the refractive index, use Fradkin for z and Wong for y
n_y_by_wong   = @(lambda) sqrt(2.09930 + 0.922683./(1 - 0.0467695*lambda.^-2) - 0.0138404*lambda.^2);
pump_crystal  = @(lambda,T) n_y_by_wong(lambda)+dn_dty1(lambda,T);
idler_crystal = @(lambda,T) n_y_by_wong(lambda)+dn_dty1(lambda,T);
signal_crystal= @(lambda,T) nz_KTP_Arie(lambda,T);

% pump_crystal  = @(lambda,T) ny_KTP_kato(lambda,T);
% idler_crystal = @(lambda,T) ny_KTP_kato(lambda,T);
% signal_crystal= @(lambda,T) nz_KTP_kato(lambda,T);

%pump, y polarized
lambda_p        = linspace(500e-9,1500e-9,N);
w_p             = 2*pi*c./lambda_p;
n_p             = pump_crystal(lambda_p*1e6,T);
k_p             = 2*pi*n_p./lambda_p;
vg_p            = gradient(k_p)./gradient(w_p); %dk/dw

%signal, z polarized
lambda_s        = lambda_p*2;
w_s             = 2*pi*c./lambda_s;
n_s             = signal_crystal(lambda_s*1e6,T);
k_s             = 2*pi*n_s./lambda_s;
vg_s            = gradient(k_s)./gradient(w_s); %dk/dw

%idler, y polarized
lambda_i        = (1./lambda_p-1./lambda_s).^-1;
w_i             = 2*pi*c./lambda_i;
n_i             = idler_crystal(lambda_i*1e6,T);
k_i             = 2*pi*n_i./lambda_i;
vg_i            = gradient(k_i)./gradient(w_i); %dk/dw

%find cutting point
[val, indx]     = min(abs((vg_i+vg_s)/2-vg_p));
lambda_s_GVM    = lambda_s(indx);


figure();
plot(lambda_i*1e9,(vg_i+vg_s)/2,'r'); hold on
plot(lambda_i*1e9,vg_p,'b');
xlabel('signal (idler) frequnecy [nm]')
ylabel('dk/d\omega [1/m]')
legend('(k_s'' + k_i'')/ 2','k_p''')
title('Group velocity matching in type II KTP (k_s'' + k_i'')/ 2 = k_p''')
x = [lambda_s_GVM/max(lambda_i)*1.1 lambda_s_GVM/max(lambda_i)]-0.1;
y = [0.4 0.4];
text_string = sprintf('cutting point at:\n %d[nm] of idler,\n %d[nm] pump', (round(lambda_s_GVM*1e9)),round(lambda_s_GVM*1e9/2));
annotation('textarrow',x,y,'String',text_string)
set(findall(gcf,'-property','FontSize'),'FontSize',16);
set(findall(gcf,'-property','LineWidth'),'LineWidth',3);

%GVM parameters:
lambda_p_GVM = lambda_s_GVM/2;
k_p_GVM      = 2*pi*ny_KTP_kato(lambda_p_GVM*1e6,T)/lambda_p_GVM;
k_s_GVM      = 2*pi*pump_crystal(lambda_s_GVM*1e6,T)/lambda_s_GVM;
k_i_GVM      = 2*pi*signal_crystal(lambda_s_GVM*1e6,T)/lambda_s_GVM;
delta_k_GVM  = k_p_GVM-k_s_GVM-k_i_GVM;
poling_GVM   = 2*pi/delta_k_GVM;


