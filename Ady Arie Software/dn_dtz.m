% dn_dtz
% Fisher and Arie, Applied Optics 42, 6661 (2003)
function dn=dn_dtz(T,lambda)
% lambda in microns
a0=9.9587e-6;
a1=9.9288e-6;
a2=-8.9603e-6;
a3=4.101e-6;
b0=-1.1882e-8;
b1=10.459e-8;
b2=-9.8136e-8;
b3=3.1481e-8;
dn=(a0+a1./lambda+a2./lambda.^2+a3./lambda.^3).*(T-25)+ (b0+b1./lambda+b2./lambda.^2+b3./lambda.^3).*(T-25).^2;