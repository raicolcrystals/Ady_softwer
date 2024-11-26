% n_ktp_5gT.m
% From Keren Applied phys lett 99 and Fisher and Arie, Applied Optics 42, 6661 (2003) for thermal
% dependence
function n=n_ktp_5gT(lambda,T)
n=n_ktp_5g(lambda)+dn_dtz(T,lambda);
 