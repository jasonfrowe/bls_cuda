import numpy as np
from scipy import stats #For Kernel Density Estimation
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import ScalarFormatter

def mhgmcmc(x,llx,beta,loglikelihood,*args,buffer=[],corbeta=1):
    "A Metropolis-Hastings MCMC with Gibbs sampler"
    
    xt=np.copy(x)                            #make a copy of our current state to the trail state
    npars=len(x)                             #number of parameters
    n=int(np.random.rand()*npars)            #random select a parameter to vary.
    
    if beta[n] <= 0:
        xt[n]+=0
    else:
        xt[n]+=np.random.normal(0.0,beta[n]) #Step 2: Generate trial state with Gibbs sampler 
        
    llxt=loglikelihood(xt,*args)             #Step 3 Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    
    alpha=min(np.exp(llxt-llx),1.0)          #Step 4 Compute the acceptance probability
    
    u=np.random.rand()                       #Step 5 generate a random number

    if u <= alpha:                           #Step 6, compare u and alpha
        xp1=np.copy(xt)                      #accept new trial
        llxp1=np.copy(llxt)
        ac=[0, n]                            #Set ac to mark acceptance
    else:
        xp1=np.copy(x)                       #reject new trial
        llxp1=np.copy(llx)
        ac=[1, n]                            #Set ac to mark rejectance
        
    return xp1,llxp1,ac;                     #return new state and log(p(x|d)) 
    
def demhmcmc(x,llx,beta,loglikelihood,*args,buffer=[],corbeta=1):
    "A Metropolis-Hastings MCMC with Gibbs sampler"
    
    nbuffer=len(buffer[:,0])
    rsamp=np.random.rand() #draw a random number to decide which sampler to use
    
    if rsamp < 0.5: #if rsamp is less than 0.5 use a Gibbs sampler 

        xt=np.copy(x)                            #make a copy of our current state to the trail state
        npars=len(x)                             #number of parameters
        n=int(np.random.rand()*npars)            #random select a parameter to vary.
    
        if beta[n] <= 0:
            xt[n]+=0
        else:
            xt[n]+=np.random.normal(0.0,beta[n]) #Step 2: Generate trial state with Gibbs sampler


    else:   #use our deMCMC sampler

        n=-1 #tell the accept array that we used the deMCMC sampler
        i1=int(np.random.rand()*nbuffer)
        i2=int(np.random.rand()*nbuffer)
        vectorjump=buffer[i1,:]-buffer[i2,:]
        xt=x+vectorjump*corbeta
    
    llxt=loglikelihood(xt, *args) #Step 3 Compute log(p(x'|d))=log(p(x'))+log(p(d|x'))
    
    alpha=min(np.exp(llxt-llx),1.0)          #Step 4 Compute the acceptance probability
    
    u=np.random.rand()                       #Step 5 generate a random number

    if u <= alpha:                           #Step 6, compare u and alpha
        xp1=np.copy(xt)                      #accept new trial
        llxp1=np.copy(llxt)
        ac=[0, n]                             #Set ac to mark acceptance
    else:
        xp1=np.copy(x)                       #reject new trial
        llxp1=np.copy(llx)
        ac=[1, n]                             #Set ac to mark rejectance
        
    xp1=np.array(xp1)
    return xp1,llxp1,ac;                     #return new state and log(p(x|d)) 

def genchain(x,beta,niter,loglikelihood,mcmcfunc,*args,buffer=[],corbeta=1): 
    "Generate Markov Chain"
    chain=[]                                  #Initialize list to hold chain values
    accept=[]                                 #Track our acceptance rate
    chain.append(x)                           #Step 1: start the chain
    accept.append((0,0))
    llx=loglikelihood(x,*args)    #pre-compute the log-likelihood for Step 3
    
    
    for i in range(0,niter):
        x,llx,ac = mcmcfunc(x,llx,beta,loglikelihood,buffer=buffer,corbeta=corbeta,*args)
        chain.append(x)
        accept.append(ac)
        
    chain=np.array(chain)                     #Convert list to array
    accept=np.array(accept)
    
    return chain, accept;
    
def plotchains(chain,label,colour,burnin,savefig=0):
    plt.figure(figsize=(12,12)) #adjust size of figure

    npar=len(chain[0,:])
    for i in range(0,npar):
        plt.subplot(npar, 1, i+1)      
        plt.plot(chain[burnin:,i],c=colour[i])  #plot parameter a
        plt.ylabel(label[i])                   #y-label

    plt.xlabel('Iteration')           #x-label

    if savefig != 0:
        plt.savefig(savefig)
    plt.show()

def plotmodels(time,data,chain,func,burnin,sol,id_to_fit,savefig=0,funcArgs=[]):
    "Plot the original model, some chains and the mean model"
    plt.figure(figsize=(10,7)) #adjust size of figure
    plt.xlabel('time')            #x-label
    plt.ylabel('f(x)/data')         #y-label
    #plt.plot(t,g,c='red')      
    plt.scatter(time,data,c='blue', s=100.0, alpha=0.8, edgecolors="none", label='data')  #plot our data
    chainlen=len(chain[:,0])   #number of chains
    for i in range(0,200):     #plot 200 random chain elements
        nchain=int(np.random.rand()*(chainlen-burnin)+burnin)  #randomly select a chain element that is not burnin
        # Recreate the full solution with the result
        sol_full = np.copy(sol)
        for i, ind in enumerate(id_to_fit):
            sol_full[ind] = chain[nchain,i]
        plotdata=func(sol_full,time,*funcArgs) #return MC-model with chain element parameters
        plt.plot(time,plotdata,c='r', alpha=0.05)                       #plot the MC-model.
    #plotdata=func(t,mm)
    #plt.plot(t,plotdata,c='r', alpha=1.0, lw=3, label='mean MC model')
    #plt.plot(t,g,c='green',lw=3, label='input model')
    plt.legend()
    
    if savefig != 0:
        plt.savefig(savefig)
    plt.show()
    
def triplot(chain,burnin,label,colour,nbin,ntick=5,savefig=0):
    "Making a Triangle Plot"

    nullfmt = NullFormatter()       # removing tick labels
    deffmt = ScalarFormatter()      # adding tick labels
    n=len(chain[1,:])               # determine number of subplots
    plt.figure(figsize=(10, 10))   # make a square figure
    wsize=0.9/n                     # size of individual panels

    prange=np.zeros(shape=(n,2))    #store range of parameters in chains 
    for i in range(0,n):
        prange[i,0]=min(chain[burnin:,i]) #find minimum values
        prange[i,1]=max(chain[burnin:,i]) #find maximum values
        if np.std(chain[burnin:,i]) == 0:  #catch potential problems when var is fixed
            prange[i,0]=prange[i,0]-1.0
            prange[i,1]=prange[i,1]+1.0
    
    jj=-1 
    for j in range(0,n):        #loop over each variable
        if np.std(chain[burnin:,j]) > 0:
            jj=jj+1
        
            ii=jj-1
            for i in range(j,n):    #loop again over each variable 
                if np.std(chain[burnin:,i]) > 0:
                    ii=ii+1
                
                    #determine panel size: left position and width
                    left, width = 0.1+jj*wsize, wsize    
                    #determine panel size: bottom position and width    
                    bottom, height = 0.9-ii*wsize, wsize     
                    rect_scatter = [left, bottom, width, height]#save panel size position    
                    axScatter = plt.axes(rect_scatter)          #set panel size
        
                    #put histogram on diagonals, scatter plot otherwise 
                    if i == j:
                
                        plt.hist(chain[burnin:,j],nbin,histtype='stepfilled', density=True, \
                         facecolor=colour[i], alpha=0.8)
                
                        if np.std(chain[burnin:,j]) > 0:
                            #make a uniform sample across the parameter range
                            x_eval = np.linspace(prange[j,0], prange[j,1], num=100) 
                            kde1 = stats.gaussian_kde(chain[burnin:,j],0.3) #Kernel Density Estimate
                            #overlay the KDE
                            plt.plot(x_eval, kde1(x_eval), 'k-', lw=3)
                
                        x1,x2,y1,y2 = plt.axis()                 
                        plt.axis((prange[j,0],prange[j,1],y1,y2))
                    else:
                        axScatter.scatter(chain[burnin:,j],chain[burnin:,i],c="black", s=1.0, \
                         alpha=0.1, edgecolors="none")
                        plt.axis((prange[j,0],prange[j,1],prange[i,0],prange[i,1]))   
            
                    #to use a sensible x-tick range, we have to do it manually.
                    dpr=(prange[j,1]-prange[j,0])/ntick         #make ntick ticks
                    rr=np.power(10.0,np.floor(np.log10(dpr)))
                    npr=np.floor(dpr/rr+0.5)*rr
                    plt.xticks(np.arange(np.floor(prange[j,0]/rr)*rr+npr, \
                     np.floor(prange[j,1]/rr)*rr,npr),rotation=30) #make ticks
        
                    if i != j:    
                        #use a sensible y-tick range, we have to do it manually
                        dpr=(prange[i,1]-prange[i,0])/ntick         #make ntick ticks
                        rr=np.power(10.0,np.floor(np.log10(dpr)))
                        npr=np.floor(dpr/rr+0.5)*rr
                        plt.yticks(np.arange(np.floor(prange[i,0]/rr)*rr+npr,\
                        np.floor(prange[i,1]/rr)*rr,npr),rotation=0) #make ticks
        
                    #default is to leave off tick mark labels
                    axScatter.xaxis.set_major_formatter(nullfmt)  
                    axScatter.yaxis.set_major_formatter(nullfmt)
        
                    #if we are on the sides, add tick and axes labels
                    if i==n-1:
                        axScatter.xaxis.set_major_formatter(deffmt)
                        plt.xlabel(label[j]) 
                    if j==0 :
                        if i > 0:
                            axScatter.yaxis.set_major_formatter(deffmt)
                            plt.ylabel(label[i]) 

    if savefig != 0:
        plt.savefig(savefig)
    plt.show()
    
    return;    

def gelmanrubin(*chain,burnin,npt):
    "Estimating PSRF"
    M=len(chain)         #number of chains
    N=chain[0].shape[0]-burnin #assuming all chains have the same size.
    npars=chain[0].shape[1] #number of parameters
    pmean=np.zeros(shape=(M,npars)) #allocate array to hold mean calculations 
    pvar=np.zeros(shape=(M,npars))  #allocate array to hold variance calculations

    
    for i in range(0,M):
        currentchain=chain[i]
        for j in range(0,npars):
            pmean[i,j]=np.mean(currentchain[burnin:,j]) #Generate means for each parameter in each chain
            pvar[i,j]=np.var(currentchain[burnin:,j])   #Generate variance for each parameter in each chain
    
    posteriormean=np.zeros(npars) #allocate array for posterior means
    for j in range(0,npars):
        posteriormean[j]=np.mean(pmean[:,j]) #calculate posterior mean for each parameter
        
    #Calculate between chains variance
    B=np.zeros(npars)
    for j in range(0,npars):
        for i in range(0,M):
            B[j]+=np.power((pmean[i,j]-posteriormean[j]),2)
    B=B*N/(M-1.0)    
    
    #Calculate within chain variance
    W=np.zeros(npars)
    for j in range(0,npars):
        for i in range(0,M):
            W[j]+=pvar[i,j]
    W=W/M 
    
    
    #Calculate the pooled variance
    V=(N-1)*W/N + (M+1)*B/(M*N)
    
    dof=npt-1 #degrees of freedom 
    Rc=np.sqrt((dof+3.0)/(dof+1.0)*V/W) #PSRF from Brooks and Gelman (1997)
    
    #Calculate Ru
    #qa=0.95
    #ru=np.sqrt((dof+3.0)/(dof+1.0)*((N-1.0)/N*W+(M+1.0)/M*qa))
    
    return Rc;

def calcacrate(accept,burnin):
    "Calculate Acceptance Rates"
    nchain=len(accept[:,0])
    print ('%s %.3f' % ('Global Acceptance Rate:',(nchain-burnin-np.sum(accept[burnin:,0]))/(nchain-burnin)))

    for j in range(max(accept[burnin:,1])+1):
        denprop=0   #this is for deMCMC
        deacrate=0  #this is for deMCMC
        
        nprop=0   #number of proposals
        acrate=0  #acceptance rate
        
        for i in range(burnin,nchain): #scan through the chain.
            if accept[i,1] == j :
                nprop=nprop+1
                acrate=acrate+accept[i,0]
            if accept[i,1] == -1 :
                denprop=denprop+1
                deacrate=deacrate+accept[i,0]
                
        print('%s Acceptance Rate %.3f' % (str(j),(nprop-acrate)/(nprop+1)))
    
    #if we have deMCMC results, report the acceptance rate.
    if denprop > 0:
        print('%s Acceptance Rate %.3f' % ('deMCMC',(denprop-deacrate)/denprop))
        
    return;
    
def betarescale(x,beta,niter,burnin,loglikelihood,mcmcfunc,*args,imax=20):
    "Calculate rescaling of beta to improve acceptance rates"
    
    alow = 0.22  #alow, ahigh define the acceptance rate range we want
    ahigh = 0.28
    
    delta=0.01  #parameter controling how fast corscale changes - from Gregory 2011.
    
    npars=len(x)   #Number of parameters
    acorsub=np.zeros(npars) 
    nacor=np.ones(npars)       #total number of accepted proposals
    nacorsub=np.ones(npars)    #total number of accepted proposals immediately prior to rescaling
    npropp=np.ones(npars)      #total number of proposals
    nproppsub=np.ones(npars)   #total number of proposals immediately prior to rescaling
    acrate=np.zeros(npars)      #current acrate 
    corscale=np.ones(npars)
    
    #inital run
    chain,accept=genchain(x,beta,niter,loglikelihood,mcmcfunc,*args) #Get a MC   
    nchain=len(chain[:,0])
    
    #calcalate initial values of npropp and nacor 
    for i in range(burnin,nchain):
        j=accept[i,1]           #get accept flag value
        npropp[j]+=1            #update total number of proposals
        nacor[j]+=1-accept[i,0] #update total number of accepted proposals
        
    #update x
    xin=chain[niter,:]  #we can continue to make chains by feeding the current state back into genchain
    
    acrate=nacor/npropp #inital acceptance rate
    
    afix=np.ones(npars)  #afix is an integer flag to indicate which beta entries need to be updated
    for i in range(0,npars):
        if acrate[i]<ahigh and acrate[i]>alow:   #we strive for an acceptance rate between alow,ahigh
            afix[i]=0    #afix=0 : update beta, afix=1 : do not update beta

    #We will iterate a maximum of imax times - avoid infinite loops
    icount=0   #counter to track iterations
    while (np.sum(afix) > 0):
        icount+=1  #track number of iterations
        
        if icount>1:
            npropp=np.copy(nproppsub)
            nacor=np.copy(nacorsub)
        nacorsub=np.ones(npars)  #reset nacorsub counts for each loop
        nproppsub=np.ones(npars) #reset nproppsub counts for each loop
        
        #Make another chain starting with xin
        betain=beta*corscale   #New beta for Gibbs sampling   
        chain,accept=genchain(xin,betain,niter,loglikelihood,mcmcfunc,*args) #Get a MC
        xin=chain[niter,:]     #Store current parameter state 
        
        for i in range(burnin,nchain): #scan through Markov-Chains and count number of states and acceptances 
            j=accept[i,1]
            if acrate[j]>ahigh or acrate[j]<alow: 
                npropp[j]+=1            #update total number of proposals
                nacor[j]+=1-accept[i,0] #update total number of accepted proposals
                nproppsub[j]+=1            #Update current number of proposals
                nacorsub[j]+=1-accept[i,0] #Update current number of accepted proposals
    
        for i in range(0,npars):  #calculate acceptance rates for each parameter that is to updated 
            if afix[i]>0:
                #calculate current acrates
                acrate[i]=nacorsub[i]/nproppsub[i]
    
                #calculate acorsub
                acorsub[i]=(nacor[i]-nacorsub[i])/(npropp[i]-nproppsub[i])
    
                #calculate corscale
                fterm = (acorsub[i]+delta)*0.75/(0.25*(1.0-acorsub[i]+delta))
                if fterm > 0.0:
                    corscale[i]=np.abs(corscale[i]*np.power(fterm ,0.5))
    
        print('Current Acceptance: ',acrate) #report acceptance rates
        for i in range(0,npars):  #check which parameters have achieved required acceptance rate
            if acrate[i]<ahigh and acrate[i]>alow:
                afix[i]=0

        if(icount>imax):   #if too many iterations, then we give up and exit
            afix=np.zeros(npars)
            print("Too many iterations: icount > imax")
    
    print('Final Acceptance: ',acrate) #report acceptance rates
    
    return corscale;
    
