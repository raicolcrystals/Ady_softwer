% dn_dty
% Fisher and Arie, Applied Optics 42, 6661 (2003)
function dn=dn_dty(lambda,T)
% lambda in microns
a0=6.2897e-6;
a1=6.3061e-6;
a2=-6.0629e-6;
a3=2.6486e-6;
b0=-0.14445e-8;
b1=2.2244e-8;
b2=-3.5770e-8;
b3=1.3470e-8;
dn=(a0+a1./lambda+a2./lambda.^2+a3./lambda.^3).*(T-25)+ (b0+b1./lambda+b2./lambda.^2+b3./lambda.^3).*(T-25).^2;