function n  = nz_KTP_Arie( lambda, T )
%Calculate refractive index of KTP along z axis
%   lambda in microns, T in Celsius.
%   dn/dT is According to Fisher and Arie, Applied Optics 42, 6661 (2003)
%   Sellmier according to K. Fradkin, A. Arie, A. Skliar, and G. Rosenman, “Tunable midinfrared source by difference frequency generation in bulk periodically poled KTiOPO4,” Appl. Phys. Lett. 74, 914–916 (1999). 

%calculate Selllmeier
A=2.127246810298;
B=1.184312171943;
C=0.051485232676;
D=0.009689556913;
E=0.660296369063;
F=100.005073661931;
n=(A+B./(1-C./lambda.^2)+E./(1-F./lambda.^2)-D.*lambda.^2).^0.5;

%calculate change from temperature
a0=9.9587e-6;
a1=9.9288e-6;
a2=-8.9603e-6;
a3=4.101e-6;
b0=-1.1882e-8;
b1=10.459e-8;
b2=-9.8136e-8;
b3=3.1481e-8;
dn=(a0+a1./lambda+a2./lambda.^2+a3./lambda.^3).*(T-25)+ (b0+b1./lambda+b2./lambda.^2+b3./lambda.^3).*(T-25).^2;

n=n+dn;
end

