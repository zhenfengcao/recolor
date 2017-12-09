# encoding: utf-8
from pylab import *
from numpy import *
from time import time

class Result:
    def __init__(self,x=None, fun=None):
        self.x = x
        self.fun = fun
    
class Tunner:
    
    def __init__(self, func, bounds, depth=20, npop=None, mutate_rate=0.5, projfunc=None, discrete_dims=None, nbears=75, npop_new=None):
        if not isinstance(bounds, ndarray):
            bounds = array(bounds,dtype=float)
        self.bounds = bounds
        self.ndim = bounds.shape[0]
        if npop is None:
            npop = min(2**self.ndim,100)
        self.npop = npop
        self.npop = self.npop + (self.npop % 2)
        if npop_new is None:
            npop_new = self.npop
        self.npop_new = npop_new
        self.depth = depth
        self.discrete_dims = discrete_dims
        if projfunc is not None:
            self.func = lambda x: func(projfunc(x))
        else:
            self.func = func
        self.strlen = self.ndim * self.depth
        self.mutate_rate = mutate_rate
        self.lower_bounds = self.bounds[:,0]
        self.upper_bounds = self.bounds[:,1]
        self.L = self.upper_bounds - self.lower_bounds
        self.R = 0.5**arange(1,self.depth+1,dtype=float)
        self.NBears = nbears
        self.nbears = nbears
        self.Losses = []
        self.init_pops()

    def shoot(self,strings=None):
        if strings is None:
            strings = self.strings
        x = self.lower_bounds + 0.5 * self.L * (1.0 + strings.dot(self.R))
        if self.discrete_dims is not None:
            x[self.discrete_dims] = x[self.discrete_dims].round()
        return x

    def init_pops(self):
        self.strings = 2 * random.randint(0,2,[self.npop,self.ndim,self.depth],dtype=int)-1
        self.losses = self.calc_losses(self.strings)
        
    def calc_losses(self, strings):
        values = self.shoot(strings)
        losses = zeros(values.shape[0])
        for n in range(values.shape[0]):
            losses[n] = self.func(values[n,:])
        return losses

    def reset(self):
        new_strings = zeros([self.npop, self.ndim, self.depth],dtype=int)
        for n in range(0, self.npop//2):
            i = random.randint(0,self.npop)
            str1 = self.strings[i,:,:]
            str2 = 2 * random.randint(0,2,[self.ndim,self.depth],dtype=int)-1
            str1, str2 = self.cross(str1,str2)
            new_strings[2*n,:,:]=str1
            new_strings[2*n+1,:,:]=str2
        new_losses = self.calc_losses(new_strings)
        ret = self.optimal
        kept = argsort(new_losses)[:self.npop-1]
        self.strings[1:,:,:] = new_strings[kept,:,:]
        self.strings[0,:,:] = ret.string
        self.losses[1:] = new_losses[kept]
        self.losses[0] = ret.fun
        self.Losses.append(ret.fun)
        self.nbears = self.NBears

    @property
    def saturated(self):
        if self.nbears>0: return False
        if min(diff(self.Losses[-5:])) > - 1e-4:
            return True
        else:
            return False

    def cross(self, str1, str2, d0=0,d1=None):
        if d1 is None: d1 = self.depth
        shape = str1.shape
        similarities = ((str1==str2).sum(1)/float(self.depth))
        break_points = random.randint(d0,d1,self.ndim) #((self.depth-1)*random.beta(1.5,1.5,self.ndim)).round().astype(int)#
        new_str1 = str1.copy()
        new_str2 = str2.copy()
        for i in range(self.ndim):
            new_str1[i, :break_points[i]] = str2[i, :break_points[i]]
            new_str2[i, :break_points[i]] = str1[i, :break_points[i]]
            if similarities[i] >=0.99:
                new_str1[i, :] = self.mutate(new_str1[i, :],mutate_rate=self.mutate_rate)
                new_str2[i, :] = self.mutate(new_str2[i, :],mutate_rate=self.mutate_rate)
        return new_str1, new_str2

    def rand(self):
        if random.random()<0.5:
            return -1
        else:
            return 1
        
    def flip(self,x):
        if x==1:
            return -1
        else:
            return 1

    def mutate(self,string, mutate_rate=None):
        if mutate_rate is None:
            mutate_rate = 0.5
        n = random.binomial(self.depth, mutate_rate)
        pos = random.randint(0, self.depth, n)
        for i in pos:
            string[i] = self.rand()
        return string

    def evolve(self):
        if self.saturated:
            self.reset()
        else: 
            new_strings = zeros([self.npop_new, self.ndim, self.depth],dtype=int)
            for n in range(self.npop_new//2):
                i, j = random.randint(0,self.npop,2)
                str1 = self.strings[i,:,:]
                str2 = self.strings[j,:,:]
                str1, str2 = self.cross(str1,str2)
                new_strings[2*n,:,:]=str1
                new_strings[2*n+1,:,:]=str2
            new_losses = self.calc_losses(new_strings)
            strings = concatenate((self.strings,new_strings),axis=0)
            losses = concatenate((self.losses,new_losses),axis=0)
            kept = argsort(losses)[:self.npop]
            self.strings = strings[kept,:,:]
            self.losses = losses[kept]
            self.Losses.append(losses[kept[0]])
        
    @property
    def optimal(self):
        ret = Result()
        idx = argmin(self.losses)
        ret.fun = self.losses[idx]
        ret.string = self.strings[idx,:,:]
        ret.x = self.shoot(ret.string)
        return ret
    
    def minimize(self, T=500,disp=False, draw=False, draw_prefix=''):
        self.T = T
        self. t = 0
        tic = time()
        for self.t in range(self.T):
            self.evolve()
            if disp:
                toc = time()
                if toc-tic>0.5:
                    print('Step %d/%d: loss=%f' % (self.t,self.T,self.optimal.fun))
                    tic = time()
            self.nbears -= 1
        if draw:
            fig=figure(1,figsize=(4,3))
            plot(self.Losses)
            xlabel('Time')
            ylabel('loss')
            if min(self.Losses)>=0:
                yscale('log')
            savefig(draw_prefix+'loss-vs-time.png',format='png',box_inches='tight')
            close('all')
        return self.optimal
        
        
    
    
