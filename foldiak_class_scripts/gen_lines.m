% gen_lines.m - generate line data

function X=gen_lines(N,batch_size,p)

sz=sqrt(N);

X=zeros(N,batch_size);

for i=1:batch_size
    
    im=zeros(sz);
    
    xi=find(rand(sz,1)<p);
    im(:,xi)=1;
    yi=find(rand(sz,1)<p);
    im(yi,:)=1;
    
    X(:,i)=reshape(im,N,1);
end
