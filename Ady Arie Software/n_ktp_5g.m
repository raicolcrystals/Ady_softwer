% n_ktp_5g.m
% From Keren Applied phys lett 99
function n=n_ktp_5g(lambda)
A=2.127246810298;
B=1.184312171943;
C=0.051485232676;
D=0.009689556913;
E=0.660296369063;
F=100.005073661931;
n=(A+B./(1-C./lambda.^2)+E./(1-F./lambda.^2)-D.*lambda.^2).^0.5;
 