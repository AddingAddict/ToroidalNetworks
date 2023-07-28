
function Mat2Cluster_KadmonHigherOrdersDelay(N,T,dt,g0Proposed,gbar0Proposed,m0,dtau,NLE,TWONS,ONStep,rep)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%timing
tic

%When to save
timeout=428400; % 4 dias 23 horas 
%timeout=43000 % bit less than 12 hs
% Seed
rng(rep,'twister');

%Delay
d=floor(dtau/dt);

%Time
nWONS=floor(TWONS/dt); %This is ON warm up steps
nT=floor(T/dt);         %This is total run  steps

%J
J=randn(N,N);

% Find the K and the J0 compatible with that and N

K=ceil(1/((g0Proposed/gbar0Proposed)^2+1/N));
J0=sqrt(g0Proposed^2+(gbar0Proposed^2)/N);


% The parameters that we finally use
g0=sqrt(1-K/N)*J0;
gbar0=-sqrt(K)*J0;




%QR
[Q, ~] = qr(rand(N));
Q=Q(:,1:NLE);

%Spectrum
LSNormalized=0;
LS=0;

%Initialize
x=(rand(N,1)-0.5)*0.1 ;

xp=zeros(N,1);

%%%%%%%%%%%%%%%%%%% WARM UP ONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:nWONS
    
    
    xp=(-x +g0/sqrt(N)*J*(x.*(x>0))+gbar0*mean(x.*(x>0))+sqrt(K)*J0*m0 )*dt+x;

    Jaco=JacobianAsymmetricKadmonHigherOrders(N,J,g0,gbar0,x)*dt;
    Q =Jaco*Q; %Evolve ON system
    
    if (mod(i,ONStep)==0)
        [Q R]=qr(Q,0); %Re-Orthonormalize
    end
    
    x=xp;
end

%%%%%%%%%%%%%%%%%%% CALCULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i=1:nT
    t=i*dt;
    xp=(-x +g0/sqrt(N)*J*(x.*(x>0))+gbar0*mean(x.*(x)>0)+sqrt(K)*J0*m0 )*dt+x;
    Jaco=JacobianAsymmetricKadmonHigherOrders(N,J,g0,gbar0,x)*dt;
    Q =Jaco*Q; %Evolve ON system
    
    if (mod(i,ONStep)==0)
        [Q R]=qr(Q,0); %Re-Orthonormalize
        LS=LS+log(diag(abs(R)));
        
        runtime=toc;
        if (runtime>timeout||sum(isnan(LS)))
            disp('leaving early')            
            break;
        end
        
        
    end
    x=xp;

    
    
end

LSNormalized=LS/t;

%%%%%%%%%%%%%%%%%%% Save OUTCOME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sN=num2str(N);
sT=num2str(T);
sdt=num2str(dt);
sg0Proposed=num2str(g0Proposed);
sgbar0Proposed=num2str(gbar0Proposed);
sm0=num2str(m0);
sdtau=num2str(dtau);
sNLE=num2str(NLE);
sTWONS=num2str(TWONS);
sONStep=num2str(ONStep);
srep=num2str(rep);



name=['-N_',sN,'-T_',sT,'-dt_',sdt,'-g0Proposed_',sg0Proposed,...
    '-gbar0Proposed_',sgbar0Proposed,'-m0_',sm0,...
    '-dtau_',sdtau,'-NLE_',sNLE,'-TWONS_',sTWONS,'-ONStep_',sONStep,...
    '-rep_',srep];


cwd=pwd;
newSubFolder=[cwd,'/Output/'];
if ~exist(newSubFolder, 'dir')
    mkdir(newSubFolder);
end
save([newSubFolder,['LS-Driven-',name],'.mat'],'LSNormalized')
toc
disp(i*dt)

end


function Jaco=JacobianAsymmetricKadmonHigherOrders(N,J,g0,gbar0,x)


Jaco=zeros(N);
DerMat=repmat((x>0)',[N,1]);

Jaco=-eye(N)+(g0/sqrt(N)*J+gbar0/N).*DerMat;




end

