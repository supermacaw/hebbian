% showrfs.m - function to show feedforward weights
%
% function showrfs(Q)
%
% Q = rf matrix

function showrfs(Q)

[M N]=size(Q);

sz=sqrt(N);

buf=1;

if floor(sqrt(M))^2 ~= M
  n=sqrt(M/2);
  m=M/n;
else
  m=sqrt(M);
  n=m;
end

array=0.5*ones(buf+n*(sz+buf),buf+m*(sz+buf));

k=1;

for j=1:m
  for i=1:n

    clim=max(abs(Q(k,:)));
    array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=...
        reshape(Q(k,:),sz,sz)/clim;
    k=k+1;
    
  end
end

imagesc(array), axis image off

