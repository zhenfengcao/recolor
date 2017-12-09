# encoding: utf-8
from pylab import *
from numpy import *
from tunner import Tunner
from PIL import Image
from pylab import figure, savefig, imshow, close

class Recolor:
    def __init__(self, source, ncolors=14, size=[50,50], save_data=True):
        self.source = source
        self.path = source.split('.')
        self.name = '.'.join(self.path[:-1])
        self.fmt = self.path[-1]
        self.ncolors = ncolors
        self.size=size
        if self.fmt =='npy':
            self.load_saved(source)
        else:
            self.load(source)
            self.get_colors(ncolors)
            if save_data:
                self.save_data()
                
    def save_data(self):
        save(self.name+'-colors.npy', self.colors)
        fig = figure(figsize=(self.colors.shape[0],1))
        colors = self.colors.reshape([1,self.colors.shape[0], 3])
        imshow(colors/255.0, aspect=1.0)
        xticks([])
        yticks([])
        savefig(self.name+'-colors.png',format='png',box_inches='tight')
        close(fig)
        
    def load_saved(self,path):
        self.colors = load(path)
        self.ncolors = self.colors.shape[0]-2

    def load(self, img_path):
        self.im = Image.open(img_path)
        self.data = array(self.im.resize([20,20])).astype(float)

    def save(self, data, img_path):
        im = Image.fromarray(data)
        im.save(img_path)
        
    def objfunc(self,xyz):
        xyz = xyz.reshape([self.ncolors, 3])
        xyz = concatenate(([[0,0,0]],xyz,[[255,255,255]]),axis=0)
        loss = sqrt(((self.XYZ - xyz)**2).sum(axis=-1)).min(axis=-1).mean()
        return loss
    
    def optimize(self, T=500):
        colors = (self.op.minimize(T,disp=True).x).round().astype(float).reshape([self.ncolors, 3])
        self.colors = concatenate(([[0,0,0]],colors,[[255,255,255]]),axis=0).astype(float)

    def get_repeats(self, data):
        return repeat(data.reshape([data.shape[0],data.shape[1],1,data.shape[-1]]),self.ncolors+2,axis=2)
        
    def get_colors(self,ncolors=14):
        self.ncolors = ncolors
        self.XYZ = self.get_repeats(self.data[:,:,0:3])
        self.bounds = [[0, 255] for i in range(self.ncolors*3)]
        self.op=Tunner(self.objfunc,self.bounds,npop=200)
        self.optimize()
    def togrey(self,rgb):
        weights = array([0.3,0.59,0.11],dtype=float)
        rgb= rgb.dot(weights)
        return rgb

    def recolor(self,img_path):
        path = img_path.split('.')
        name = '.'.join(path[:-1])
        fmt = path[-1]
        im = Image.open(img_path)
        im=im.convert('RGBA')
        data = array(im).astype(float)
        XYZ = self.get_repeats(data[:,:,0:3])
        weights = array([0.3,0.59,0.11],dtype=float)
      
        color_inds = sqrt(((XYZ* weights - self.colors * weights)**2).sum(axis=-1)).argmin(axis=-1)
        new_data0 = data.copy()
        
        for i in range(0, 2+self.ncolors):
            ids = where(color_inds == i)
            if len(ids[0])>0:
                new_data0[ids[0],ids[1],:3] = self.colors[color_inds[ids],:3] 

        new_data0 = (new_data0-new_data0.min())/(new_data0.max()-new_data0.min())*255
        
        new_data0 = new_data0.astype('uint8')
        self.save(new_data0,name+'-recolored'+'.png')
        
        
        
        
        
        
        
