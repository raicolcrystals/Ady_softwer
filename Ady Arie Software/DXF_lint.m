% DXF_lint.m
%
% Sets line type (internal use only!)
%
function DXF_lint(fid,linetype)

fprintf(fid,'6\n');
switch linetype
case 1
  fprintf(fid,'CONTINUOUS\n');
case 2
  fprintf(fid,'HIDDEN2\n');
case 3
  fprintf(fid,'CENTER2\n');
case 4
  fprintf(fid,'DOT2\n');
otherwise
  fprintf(fid,'CONTINUOUS\n');
end
