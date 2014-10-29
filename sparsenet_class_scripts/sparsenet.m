% sparsenet.m - Olshausen & Field sparse coding algorithm
% 
% Before running you must first load the training data array IMAGES

num_trials=10000;
batch_size=100;

[imsize imsize num_images]=size(IMAGES);
BUFF=4;

% number of outputs
M=64;

% number of inputs
N=64;
sz=sqrt(N);

% initialize basis functions (comment out these lines if you wish to 
% pick up where you left off)
Phi=randn(N,M);
Phi=Phi*diag(1./sqrt(sum(Phi.*Phi)));

% learning rate (start out large, then lower as solution converges)
eta = 2.0;

% lambda
lambda = 0.1;

a_var=ones(M,1);
var_eta=.1;

I=zeros(N,batch_size);

display_every=5;
display_network(Phi,a_var);

for t=1:num_trials
    
    % choose an image for this batch

    imi=ceil(num_images*rand);

    % extract subimages at random from this image to make data array I

    for i=1:batch_size
        r=BUFF+ceil((imsize-sz-2*BUFF)*rand);
        c=BUFF+ceil((imsize-sz-2*BUFF)*rand);
        I(:,i)=reshape(IMAGES(r:r+sz-1,c:c+sz-1,imi),N,1);
    end

    % calculate coefficients for these data via LCA

    ahat = sparsify(I,Phi,lambda);

    % update bases

    dPhi = ... %  learning rule here
    Phi = Phi + dPhi;
    Phi=Phi*diag(1./sqrt(sum(Phi.*Phi))); % normalize bases

    % accumulate activity statistics
    
    a_var=(1-var_eta)*a_var + var_eta*mean(ahat.^2,2);

    % display

    if (mod(t,display_every)==0)
        display_network(Phi,a_var);
    end

end
