%  DXF_pinc function
%
%  Internal use only
%
%  Increments pointer and outputs as a hex value
%
function DXF_pinc(fid)
global dxfhandle

fprintf(fid,'5\n');
hexstr=sprintf('%6x',dxfhandle);
fprintf(fid,'%s\n',upper(hexstr));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% changed by Alon B. 8/6/2005
if (dxfhandle==7917)
    dxfhandle=7928; 
else
    dxfhandle=dxfhandle+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(dxfhandle==55)
  dxfhandle=dxfhandle+1;
end