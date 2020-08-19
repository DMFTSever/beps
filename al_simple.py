# Python implementation of the Boson Exchange Parquet Solver (BEPS) for a quantum impurity model (atomic limit), cite as arXiv:2008.04184
# Copyright (C) 2020 by Friedrich Krien <krien@ifp.tuwien.ac.at>

import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import csv
from numpy.linalg import inv
import glob
from matplotlib.ticker import FormatStrFormatter
import math
from scipy.optimize import broyden1,broyden2
from numpy import linalg as eigensystem
#from mpltools import color

plt.rcParams["mathtext.fontset"] = "cm"
cmap = plt.get_cmap('bwr')

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
  def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    self.midpoint = midpoint
    colors.Normalize.__init__(self, vmin, vmax, clip)

  def __call__(self, value, clip=None):
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def minmax_real(matrix_list,nu_min_,nu_max_,om):
  lmin = []
  lmax = []
  for mat in matrix_list:
    lmin.append(mat[nu_g4(nu_min_):nu_g4(nu_max_), nu_g4(nu_min_):nu_g4(nu_max_)].real.min())
    lmax.append(mat[nu_g4(nu_min_):nu_g4(nu_max_), nu_g4(nu_min_):nu_g4(nu_max_)].real.max())
  return (min(lmin),max(lmax))

def plot_vertex_real(vtx,nu_min_,nu_max_,om,elev_min_=0,elev_max_=0,mid_val_zero=True,title='',filename='',notitle=False):
  if elev_min_==0 and elev_max_==0: 
    elev_min = vtx[nu_g4(nu_min_):nu_g4(nu_max_), nu_g4(nu_min_):nu_g4(nu_max_)].real.min()
    elev_max = vtx[nu_g4(nu_min_):nu_g4(nu_max_), nu_g4(nu_min_):nu_g4(nu_max_)].real.max()
  else:
    elev_min=elev_min_
    elev_max=elev_max_
  if mid_val_zero == True: mid_val = 0.0
  else: mid_val = (elev_min + elev_max)/2.
  pdf = plt.figure()
  if notitle:
    plt.title('')
  else:
      plt.title(title, fontdict = {'fontsize' : 40})
  X = np.array(range(nu_min_,nu_max_+1))-0.5
  Y = np.array(range(nu_min_,nu_max_+1))-0.5
  plt.pcolormesh(X,Y,vtx[nu_g4(nu_min_):nu_g4(nu_max_+1), nu_g4(nu_min_):nu_g4(nu_max_+1)].real,cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
  cbar = plt.colorbar(format='%.2f')
  plt.tick_params(labelsize=20)
  cbar.ax.tick_params(labelsize=20)
  plt.xticks([-10,0,10])
  plt.yticks([-10,0,10])
  plt.show()
  if filename=='':
    pdf.savefig('W'+str(om)+'_'+title+'.pdf', pad_inches=0, dpi=200)
  else:
    pdf.savefig('W'+str(om)+'_'+filename+'.pdf', pad_inches=0, dpi=200)
  plt.close(pdf)
  return (elev_min,elev_max)


#################
U          =  1.0 #do
U_ch       =   +U #not    
U_sp       =   -U #change 
U_si       = 2.*U #this 
beta       =   2. #values of U*beta >= 2 are hard and require mixing and many iterations. Large U*beta requires a nonlinear root-finder (Broyden, Pulay, ...)
delta_beta =   0. #parameter for annealing from a higher temperature
#################

#Matsubara cutoffs
N_nu = 24 #fermionic 
N_om = 12 #bosonic

nu_min = -N_nu//2
nu_max = +N_nu//2
om_min = 0
om_max = +N_om

om_min_=0
om_max_=N_om/2
om_inc =1

plot_pause = 0.1

parquet_approximation=False
initial_guess = 0 #0=non-interacting limit,1=firr,2="guess" the exact solution (can be used to verify that it corresponds to a stable fixed point)

#various flags for consistency checks
check_bse = False #Plot MBE vertices M
check_phi = False #Plot reducible vertices of parquet decomposition
check_mbe = False #Plot reducible vertices of BEPS
check_lambda_firr = False #Plot fully irr. vertex of parquet
check_threeleg_firr = False #Plot threeleg vertex which arises from fully irr. vertex of parquet (+-1 + sum Lambda^firr gg)
check_gamma_consistency = False #Verify the consistency of the exact solution for the vertices

def psum():
  return 2. + 2.*math.exp(-beta*U/2.)

def nu_g4(w_): #four-point
  return int(w_+N_nu//2)

def om_g4(w_): #four-point
  return int(w_)

def nu_m(nu):
  return (2.*nu+1.)*math.pi/beta

def om_m(om):
  return 2.*om*math.pi/beta

def delta(om):
  if om==0:
    return 1.
  else:
    return 0.

def sigma(nu):
  return U/2. + U*U/(4.j*nu_m(nu))

def g(nu):
  return 1. / ( 1.j * nu_m(nu) - U*U / ( 4.j * nu_m(nu) ) )

def g0(nu):
  return 1. / ( 1.j * nu_m(nu) + U/2. )

#def g0_semicircle(nu):
#  return 1. / ( 1.j * nu_m(nu) + U/2. )

def chi_ch(om):
  return -2.*beta*math.exp(-beta*U/2.)*delta(om)/psum()

def chi_sp(om):
  return -2.*beta*delta(om)/psum()

def chi_si(om):
  return -beta*math.exp(-beta*U/2.)*delta(om)/psum()

def pi_ch(om):
  return (chi_ch(om)/2.) / (1. + U_ch * chi_ch(om) / 2.)

def pi_sp(om):
  return (chi_sp(om)/2.) / (1. + U_sp * chi_sp(om) / 2.)

def pi_si(om):
  return chi_si(om)      / (1. + U_si * chi_si(om) / 2.)



#################
#Functor classes#
#################

class w():
  def ch(self,om):
    return U_ch + U_ch * chi_ch(om) * U_ch/2.
    
  def sp(self,om):
    return U_sp + U_sp * chi_sp(om) * U_sp/2.

  def si(self,om):
    return U_si + U_si * chi_si(om) * U_si/2.



class thunstroem(): #analytical solution for the vertices of the atomic limit, PRB 98, 235107 (2018)
  def __init__(self):
    self.Acal0=np.zeros(4,dtype=np.complex_)
    self.Bcal0=np.zeros(4,dtype=np.complex_)
    self.Bcal1=np.zeros(4,dtype=np.complex_)
    self.Bcal2=np.zeros(4,dtype=np.complex_)
    self.Alat= np.zeros(4,dtype=np.complex_)
    self.Blat= np.zeros(4,dtype=np.complex_)

    self.Acal0[0] = 1.
    self.Acal0[1] = 1.
    self.Acal0[2] = 1./2.
    self.Acal0[3] = -1./2.

    self.Bcal0[0] = 1.
    self.Bcal0[1] = 1.
    self.Bcal0[2] = 1./2.
    self.Bcal0[3] = -1./2.

    self.Bcal1[0] = 1.j
    self.Bcal1[1] = 1.
    self.Bcal1[2] = 1.j/math.sqrt(2.)
    self.Bcal1[3] = 0.

    self.Bcal2[0] = 1.
    self.Bcal2[1] = 1.j
    self.Bcal2[2] = 1./math.sqrt(2.)
    self.Bcal2[3] = 0.

    self.Alat[0] = +(U/2.) * math.sqrt(3.)
    self.Alat[1] = 1.j*U/2.
    self.Alat[2] = 0.
    self.Alat[3] = 1.j*U/2.

    self.Blat[0] = +(U*U/4.) * ( 3.*math.exp(+U*beta/2.) - 1. ) / ( math.exp(+U*beta/2.) + 1. )#corresponds to square of B_r in Table 1 of Thunstroem paper!
    self.Blat[1] = +(U*U/4.) * ( 3.*math.exp(-U*beta/2.) - 1. ) / ( math.exp(-U*beta/2.) + 1. )#corresponds to square of B_r in Table 1 of Thunstroem paper!
    self.Blat[2] = +(U*U/4.) * ( 3.*math.exp(+U*beta/2.) - 1. ) / ( math.exp(+U*beta/2.) + 1. )#corresponds to square of B_r in Table 1 of Thunstroem paper!
    self.Blat[3] = 0.

  def Clat(self,om,flavor):
    if flavor==0:
      return + delta(om) * (U*beta/2.) / ( 1. + math.exp(+U*beta/2.) )
    else:
      return - delta(om) * (U*beta/2.) / ( 1. + math.exp(-U*beta/2.) )

  def Dlat(self,om,flavor):
    return (U*U/4.) * ( 1. + self.Clat(om,flavor) )/( 1. - self.Clat(om,flavor) )
  
  def a0(self,nu,om,flavor):
    return (beta/2.) * ( nu_m(nu) * ( nu_m(nu) + om_m(om) ) - self.Alat[flavor]*self.Alat[flavor] ) \
            / ( ( nu_m(nu)*nu_m(nu) + U*U/4. ) * ( ( nu_m(nu)+om_m(om) )*( nu_m(nu)+om_m(om) ) + U*U/4. ) )

  def b0(self,nu,om,flavor):
    return (beta/2.) * ( nu_m(nu) * ( nu_m(nu) + om_m(om) ) - self.Blat[flavor] ) \
            / ( ( nu_m(nu)*nu_m(nu) + U*U/4. ) * ( ( nu_m(nu)+om_m(om) )*( nu_m(nu)+om_m(om) ) + U*U/4. ) )

  def b1(self,nu,om,flavor):
    return self.Bcal1[flavor] * math.sqrt( U * (1. - self.Clat(om,flavor) ) ) * ( nu_m(nu) * ( nu_m(nu) + om_m(om) ) - self.Dlat(om,flavor) ) \
            / ( ( nu_m(nu)**2 + U*U/4. ) * ( ( nu_m(nu)+om_m(om) )**2 + U*U/4. ) )

  def b2(self,nu,om,flavor):
    return self.Bcal2[flavor] * math.sqrt( U*U*U/4. ) * math.sqrt( ( U*U / (1. - self.Clat(om,flavor) ) ) + om_m(om)*om_m(om) ) \
            / ( ( nu_m(nu)**2 + U*U/4. ) * ( ( nu_m(nu)+om_m(om) )**2 + U*U/4. ) )

  def chi(self,nu1,nu2,om,flavor):
    return - self.a0(nu1,om,flavor) * ( delta(nu1-nu2) - delta(nu1 + nu2 + om + 1) ) \
           - self.b0(nu1,om,flavor) * ( delta(nu1-nu2) + delta(nu1 + nu2 + om + 1) ) \
           - self.b1(nu1,om,flavor)*self.b1(nu2,om,flavor) - self.b2(nu1,om,flavor)*self.b2(nu2,om,flavor)

  def gamma(self,nu1,nu2,om_in,flavor): #Thunstroem Eq. 19
    if (flavor<2): om=om_in
    else: om=-om_in#Thunstroem paper defined pp-channels with a minus sign so they can use the same formulas for ph and pp
    if (flavor==0): flag=+1
    if (flavor==1): flag=-1
    if (flavor==2): flag=+1
    if (flavor<3):
      return  ( \
            (beta/2.) * (self.Alat[flavor] * self.Alat[flavor] / self.Acal0[flavor]) * ( nu_m(nu1)*nu_m(nu1) + U*U/4. ) * ( ( nu_m(nu1)+om_m(om) )*( nu_m(nu1)+om_m(om) ) + U*U/4. ) \
            * ( delta(nu1-nu2) - delta(nu1 + nu2 + om + 1) ) \
            / ( ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) - self.Alat[flavor]*self.Alat[flavor] ) * ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) ) ) \
            ) \
            + ( \
            (beta/2.) * (self.Blat[flavor] / self.Bcal0[flavor]) * ( nu_m(nu1)*nu_m(nu1) + U*U/4. ) \
            * ( ( nu_m(nu1)+om_m(om) )*( nu_m(nu1)+om_m(om) ) + U*U/4. ) \
            * ( delta(nu1-nu2) + delta(nu1 + nu2 + om + 1) ) \
            / ( ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) - self.Blat[flavor] ) * ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) ) ) \
            ) \
            - ( \
            U*(np.abs(self.Bcal2[flavor])*np.abs(self.Bcal2[flavor])/(self.Bcal0[flavor]*self.Bcal0[flavor]))*(U*U/4.)*((U*U/4.)*(4.*self.Blat[flavor]/(U*U)+1.)*(4.*self.Blat[flavor]/(U*U)+1.)+om_m(om)*om_m(om)) \
            / ( U * np.tan((beta/4.)*(np.sqrt(4.*self.Blat[flavor]+om_m(om)*om_m(om))+om_m(om)))/np.sqrt(4.*self.Blat[flavor]+om_m(om)*om_m(om)) + flag ) \
            / ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) - self.Blat[flavor] ) \
            / ( nu_m(nu2) * ( nu_m(nu2) + om_m(om) ) - self.Blat[flavor] ) \
            ) \
            - (self.Bcal1[flavor]*self.Bcal1[flavor]/(self.Bcal0[flavor]*self.Bcal0[flavor]))*U
    else:
      return  ( \
            (beta/2.) * (self.Alat[flavor] * self.Alat[flavor] / self.Acal0[flavor]) * ( nu_m(nu1)*nu_m(nu1) + U*U/4. ) * ( ( nu_m(nu1)+om_m(om) )*( nu_m(nu1)+om_m(om) ) + U*U/4. ) \
            * ( delta(nu1-nu2) - delta(nu1 + nu2 + om + 1) ) \
            / ( ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) - self.Alat[flavor]*self.Alat[flavor] ) * ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) ) ) \
            ) \
            + ( \
            (beta/2.) * (self.Blat[flavor] / self.Bcal0[flavor]) * ( nu_m(nu1)*nu_m(nu1) + U*U/4. ) \
            * ( ( nu_m(nu1)+om_m(om) )*( nu_m(nu1)+om_m(om) ) + U*U/4. ) \
            * ( delta(nu1-nu2) + delta(nu1 + nu2 + om + 1) ) \
            / ( ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) - self.Blat[flavor] ) * ( nu_m(nu1) * ( nu_m(nu1) + om_m(om) ) ) ) \
            )

  def vtx_ph(self,nu1,nu2,om,flavor):
    return ( self.chi(nu1,nu2,om,flavor) - beta * g(nu1) * g(nu1 + om) * delta(nu1-nu2) ) / (g(nu1) * g(nu1 + om) * g(nu2) * g(nu2 + om))

  def vtx_pp(self,nu1,nu2,om,flavor):
    return ( self.vtx_ph(nu1,nu2,om-nu1-nu2-1,0) - (3. - 4. * delta(flavor-1) )*self.vtx_ph(nu1,nu2,om-nu1-nu2-1,1) )/2.

  def ch(self,nu1,nu2,om):
    return self.vtx_ph(nu1,nu2,om,0)

  def sp(self,nu1,nu2,om):
    return self.vtx_ph(nu1,nu2,om,1)

  def si(self,nu1,nu2,om):
    return self.vtx_pp(nu1,nu2,om,0)

  def tr(self,nu1,nu2,om):
    return self.vtx_pp(nu1,nu2,om,1)

  def firr_ch(self,nu1,nu2,om):
    return self.gamma(nu1,nu2,om,0) - 0.5*self.gamma(nu1,nu1+om,nu2-nu1,0) - 1.5*self.gamma(nu1,nu1+om,nu2-nu1,1) \
        + 0.5*self.gamma(nu1,nu2,nu1+nu2+om+1,2) + 1.5*self.gamma(nu1,nu2,nu1+nu2+om+1,3) - 2.*self.ch(nu1,nu2,om)

  def firr_sp(self,nu1,nu2,om):
    return self.gamma(nu1,nu2,om,1) - 0.5*self.gamma(nu1,nu1+om,nu2-nu1,0) + 0.5*self.gamma(nu1,nu1+om,nu2-nu1,1) \
        - 0.5*self.gamma(nu1,nu2,nu1+nu2+om+1,2) + 0.5*self.gamma(nu1,nu2,nu1+nu2+om+1,3) - 2.*self.sp(nu1,nu2,om)

  def firr_si(self,nu1,nu2,om):
    return 0.5*self.firr_ch(nu1,nu2,om-nu1-nu2-1) - 1.5*self.firr_sp(nu1,nu2,om-nu1-nu2-1) 

  def firr_tr(self,nu1,nu2,om):
    return 0.5*self.firr_ch(nu1,nu2,om-nu1-nu2-1) + 0.5*self.firr_sp(nu1,nu2,om-nu1-nu2-1)

  def phi_ch(self,nu1,nu2,om):
    return self.ch(nu1,nu2,om) - self.gamma(nu1,nu2,om,0)

  def phi_sp(self,nu1,nu2,om):
    return self.sp(nu1,nu2,om) - self.gamma(nu1,nu2,om,1)

  def phi_si(self,nu1,nu2,om):
    return self.si(nu1,nu2,om) - self.gamma(nu1,nu2,om,2)

  def phi_tr(self,nu1,nu2,om):
    return self.tr(nu1,nu2,om) - self.gamma(nu1,nu2,om,3)


class mbe():
  def __init__(self,th,nb):
    self.nb = nb
    self.th = th

  def ch(self,nu1,nu2,om):
    return self.th.phi_ch(nu1,nu2,om) - self.nb.ch(nu1,nu2,om) + U_ch

  def sp(self,nu1,nu2,om):
    return self.th.phi_sp(nu1,nu2,om) - self.nb.sp(nu1,nu2,om) + U_sp

  def si(self,nu1,nu2,om):
    return self.th.phi_si(nu1,nu2,om) - self.nb.si(nu1,nu2,om) + U_si

  def tr(self,nu1,nu2,om):
    return self.th.phi_tr(nu1,nu2,om)



class hedin():
  def dgdnu(self,nu,om):
    return ( g(nu+om) - g(nu) ) / ( 1.j * om_m(om) )
  
  def dgdmu(self,nu):
    return -( ( -beta*U*math.exp(-beta*U/2.) / ( -nu_m(nu)*nu_m(nu) - U*U/4. ) ) + (psum()/2.)/(( 1.j*nu_m(nu) + U/2. )**2) + (psum()/2.)/(( 1.j*nu_m(nu) - U/2. )**2) ) / psum()
  
  def dgdh(self,nu):
    return -( ( beta*U / ( -nu_m(nu)*nu_m(nu) - U*U/4. ) ) + (psum()/2.)/(( 1.j*nu_m(nu) + U/2. )**2) + (psum()/2.)/(( 1.j*nu_m(nu) - U/2. )**2) ) / psum()
  
  def g3pp(self,nu,om):
    return ( -math.exp(-beta*U/2.)/psum() ) * \
       ( ( math.exp(beta*U/2.) + 1. ) * ( ( 1./ ( (1.j*nu_m(nu) + U/2.)*(1.j*om_m(om) - 1.j*nu_m(nu) - U/2.) ) ) + \
       ( 1. / ( (1.j*nu_m(nu) -  U/2.)*(1.j*om_m(om) - 1.j*nu_m(nu) + U/2.) ) ) ) \
       + ( ( delta(om) * beta * U ) / ( (1.j*nu_m(nu) + U/2.) * (1.j*nu_m(nu) - U/2.) ) ) )

  def ch(self,nu,om):
    if om==0:
      return -self.dgdmu(nu)    / ( g(nu) * g(nu)      * ( 1. + U_ch*chi_ch(om)/2. ) )
    else:
      return -self.dgdnu(nu,om) / ( g(nu) * g(nu+om)   * ( 1. + U_ch*chi_ch(om)/2. ) )

  def sp(self,nu,om):
    if om==0:
      return -self.dgdh(nu)     / ( g(nu) * g(nu)      * ( 1. + U_sp*chi_sp(om)/2. ) )
    else:
      return -self.dgdnu(nu,om) / ( g(nu) * g(nu+om)   * ( 1. + U_sp*chi_sp(om)/2. ) )
    
  def si(self,nu,om):
    return self.g3pp(nu,om)     / ( g(nu) * g(om-nu-1) * ( 1. + U_si*chi_si(om)/2. ) )



class nabla():
  def __init__(self,w,l):
    self.w = w
    self.l = l

  def ch(self,nu1,nu2,om):
#    return self.l.ch(nu1,om) * self.w.ch(om) * self.l.ch(nu2,om)
    return self.l.ch(nu1,om) * self.w.ch(om) * self.l.ch(nu2+om,-om)

  def sp(self,nu1,nu2,om):
#    return self.l.sp(nu1,om) * self.w.sp(om) * self.l.sp(nu2,om)
    return self.l.sp(nu1,om) * self.w.sp(om) * self.l.sp(nu2+om,-om)

  def si(self,nu1,nu2,om):
    return self.l.si(nu1,om) * self.w.si(om) * self.l.si(nu2,om)

  def vph_ch(self,nu1,nu2,om):
    return - ( self.ch(nu1,nu1+om,nu2-nu1) + 3.*self.sp(nu1,nu1+om,nu2-nu1) ) / 2.

  def vph_sp(self,nu1,nu2,om):
    return - ( self.ch(nu1,nu1+om,nu2-nu1) -    self.sp(nu1,nu1+om,nu2-nu1) ) / 2.

  def pp_ch(self,nu1,nu2,om):
    return + self.si(nu1,nu2,om + nu1 + nu2 + 1)/2.

  def pp_sp(self,nu1,nu2,om):
    return - self.si(nu1,nu2,om + nu1 + nu2 + 1)/2.

  def sbe_ch(self,nu1,nu2,om):
    return self.ch(nu1,nu2,om) + self.vph_ch(nu1,nu2,om) + self.pp_ch(nu1,nu2,om) - 2.*U_ch

  def sbe_sp(self,nu1,nu2,om):
    return self.sp(nu1,nu2,om) + self.vph_sp(nu1,nu2,om) + self.pp_sp(nu1,nu2,om) - 2.*U_sp

  def sbe_pp(self,nu1,nu2,om,flavor):
    return ( self.sbe_ch(nu1,nu2,om-nu1-nu2-1) - (3. - 4. * delta(flavor-1) )*self.sbe_sp(nu1,nu2,om-nu1-nu2-1) )/2.

  def sbe_si(self,nu1,nu2,om):
    return self.sbe_pp(nu1,nu2,om,0)
  
  def sbe_tr(self,nu1,nu2,om):
    return self.sbe_pp(nu1,nu2,om,1)


class mbe_vtx():
  def __init__(self,mb):
    self.mb = mb

  def ch(self,nu1,nu2,om):
    return self.mb.ch(nu1,nu2,om)

  def sp(self,nu1,nu2,om):
    return self.mb.sp(nu1,nu2,om)

  def si(self,nu1,nu2,om):
    return self.mb.si(nu1,nu2,om)

  def tr(self,nu1,nu2,om):
    return self.mb.tr(nu1,nu2,om)

  def vph_ch(self,nu1,nu2,om):
    return - ( self.ch(nu1,nu1+om,nu2-nu1) + 3.*self.sp(nu1,nu1+om,nu2-nu1) ) / 2.

  def vph_sp(self,nu1,nu2,om):
    return - ( self.ch(nu1,nu1+om,nu2-nu1) -    self.sp(nu1,nu1+om,nu2-nu1) ) / 2.

  def pp_ch(self,nu1,nu2,om):
    return + self.si(nu1,nu2,om + nu1 + nu2 + 1)/2. + 3.*self.tr(nu1,nu2,om + nu1 + nu2 + 1)/2.

  def pp_sp(self,nu1,nu2,om):
    return - self.si(nu1,nu2,om + nu1 + nu2 + 1)/2. +    self.tr(nu1,nu2,om + nu1 + nu2 + 1)/2.


class threeleg_uirr():
  def __init__(self,nb,th):
    self.nb = nb
    self.th = th
  
  def vtx_ch(self,nu1,nu2,om):
    return th.ch(nu1,nu2,om) - nb.sbe_ch(nu1,nu2,om)

  def vtx_sp(self,nu1,nu2,om):
    return th.sp(nu1,nu2,om) - nb.sbe_sp(nu1,nu2,om)

  def vtx_si(self,nu1,nu2,om):
    return th.si(nu1,nu2,om) - nb.sbe_si(nu1,nu2,om)

  def vtx_tr(self,nu1,nu2,om):
    return th.tr(nu1,nu2,om) - nb.sbe_tr(nu1,nu2,om)

  def ch(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_ch(nu1,nu2,om)*g(nu2)*g(nu2+om) 
    return 1. + result/beta

  def sp(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_sp(nu1,nu2,om)*g(nu2)*g(nu2+om) 
    return 1. + result/beta

  def si(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_si(nu1,nu2,om)*g(nu2)*g(om-nu2-1) 
    return -1. + result/(2.*beta)

  def tr(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_tr(nu1,nu2,om)*g(nu2)*g(om-nu2-1) 
    return -1. + result/(2.*beta)


class threeleg_firr():
  def __init__(self,th):
    self.th = th
  
  def vtx_ch(self,nu1,nu2,om):
    return th.firr_ch(nu1,nu2,om)-U_ch

  def vtx_sp(self,nu1,nu2,om):
    return th.firr_sp(nu1,nu2,om)-U_sp

  def vtx_si(self,nu1,nu2,om):
    return th.firr_si(nu1,nu2,om)-U_si

  def vtx_tr(self,nu1,nu2,om):
    return th.firr_tr(nu1,nu2,om)

  def ch(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_ch(nu1,nu2,om)*g(nu2)*g(nu2+om) 
    return 1. + result/beta

  def sp(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_sp(nu1,nu2,om)*g(nu2)*g(nu2+om) 
    return 1. + result/beta

  def si(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_si(nu1,nu2,om)*g(nu2)*g(om-nu2-1) 
    return -1. + result/(2.*beta)

  def tr(self,nu1,om):
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += self.vtx_tr(nu1,nu2,om)*g(nu2)*g(om-nu2-1) 
    return -1. + result/(2.*beta)



class parquet3():
  def __init__(self,g_,nb_,mb_,fi_):
#    self.nb=nabla()
#    self.fi=firr()
    self.g=g_
    self.nb=nb_
    self.mb=mb_
    self.fi=fi_

  def ch(self,nu,om):
    if (nu==nu_min) or (nu==nu_max): return 1.
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += ( self.mb.vph_ch(nu,nu2,om) + self.mb.pp_ch(nu,nu2,om) + self.mb.ch(nu,nu2,om) ) * self.g.value(nu2) * self.g.value(nu2+om) 
      result += ( self.nb.vph_ch(nu,nu2,om) + self.nb.pp_ch(nu,nu2,om) - 2.*U_ch )               * self.g.value(nu2) * self.g.value(nu2+om) 
    return self.fi.ch(nu,om) + result/beta

  def sp(self,nu,om): 
    if (nu==nu_min) or (nu==nu_max): return 1.
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += ( self.mb.vph_sp(nu,nu2,om) + self.mb.pp_sp(nu,nu2,om) + self.mb.sp(nu,nu2,om) ) * self.g.value(nu2) * self.g.value(nu2+om) 
      result += ( self.nb.vph_sp(nu,nu2,om) + self.nb.pp_sp(nu,nu2,om) - 2.*U_sp )               * self.g.value(nu2) * self.g.value(nu2+om) 
    return self.fi.sp(nu,om) + result/beta

  def si(self,nu,om): 
    if (nu==nu_min) or (nu==nu_max): return -1.
    result = 0.0
    for nu2 in range(nu_min,nu_max):
      result += ( self.mb.ch(nu,nu2,om-nu-nu2-1) - 3.*self.mb.sp(nu,nu2,om-nu-nu2-1) + self.mb.si(nu,nu2,om) ) * self.g.value(nu2) * self.g.value(om-nu2-1) 
      result += ( self.nb.ch(nu,nu2,om-nu-nu2-1) - 3.*self.nb.sp(nu,nu2,om-nu-nu2-1) - U_ch + 3.*U_sp )        * self.g.value(nu2) * self.g.value(om-nu2-1) 
    return self.fi.si(nu,om) + result/(2.*beta)

class kernel():
  def __init__(self,nb_,mb_,firr_):
#    self.nb=nabla()
#    self.fi=firr()
#    self.g=g_
    self.nb=nb_
    self.mb=mb_
    self.firr=firr_

  def ch(self,nu1,nu2,om):
    result =  self.firr.ch(nu1,nu2,om)
    result += self.mb.vph_ch(nu1,nu2,om) + self.mb.pp_ch(nu1,nu2,om)
    result += self.nb.vph_ch(nu1,nu2,om) + self.nb.pp_ch(nu1,nu2,om) - 2.*U_ch
    return result 

  def sp(self,nu1,nu2,om):
    result =  self.firr.sp(nu1,nu2,om)
    result += self.mb.vph_sp(nu1,nu2,om) + self.mb.pp_sp(nu1,nu2,om)
    result += self.nb.vph_sp(nu1,nu2,om) + self.nb.pp_sp(nu1,nu2,om) - 2.*U_sp
    return result 

  def si(self,nu1,nu2,om):
    result =  self.firr.si(nu1,nu2,om)
    result += self.mb.ch(nu1,nu2,om-nu1-nu2-1)/2. - 3.*self.mb.sp(nu1,nu2,om-nu1-nu2-1)/2.
    result += self.nb.ch(nu1,nu2,om-nu1-nu2-1)/2. - 3.*self.nb.sp(nu1,nu2,om-nu1-nu2-1)/2.
    result += self.mb.ch(nu1,om-nu2-1,nu2-nu1)/2. - 3.*self.mb.sp(nu1,om-nu2-1,nu2-nu1)/2.
    result += self.nb.ch(nu1,om-nu2-1,nu2-nu1)/2. - 3.*self.nb.sp(nu1,om-nu2-1,nu2-nu1)/2.
    result += - U_ch + 3.*U_sp
    return result 

  def tr(self,nu1,nu2,om):
    result =    self.firr.tr(nu1,nu2,om)
    result += + self.mb.ch(nu1,nu2,om-nu1-nu2-1)/2. + self.mb.sp(nu1,nu2,om-nu1-nu2-1)/2.
    result += + self.nb.ch(nu1,nu2,om-nu1-nu2-1)/2. + self.nb.sp(nu1,nu2,om-nu1-nu2-1)/2.
    result += - self.mb.ch(nu1,om-nu2-1,nu2-nu1)/2. - self.mb.sp(nu1,om-nu2-1,nu2-nu1)/2.
    result += - self.nb.ch(nu1,om-nu2-1,nu2-nu1)/2. - self.nb.sp(nu1,om-nu2-1,nu2-nu1)/2.
    return result 


###################
#Container classes#
###################

class hedin_container():
  def __init__(self):
    self.hedin_ch = np.zeros( (N_nu,N_om) , dtype=complex )
    self.hedin_sp = np.zeros( (N_nu,N_om) , dtype=complex )
    self.hedin_si = np.zeros( (N_nu,N_om) , dtype=complex )

  def value(self,nu,om,flavor):
    if om<0:
      return np.conjugate( self.value(-nu-1,-om,flavor) )
    else:
      if (nu<nu_min) or (nu>nu_max-1) or (om>om_max-1):
        if flavor == 2: return -1.
        else:           return +1.
      else:
        if flavor==0: return self.hedin_ch[ nu+N_nu//2,om ] 
        if flavor==1: return self.hedin_sp[ nu+N_nu//2,om ] 
        if flavor==2: return self.hedin_si[ nu+N_nu//2,om ] 
  
  def ch(self,nu,om): return self.value( nu,om,0 )
  def sp(self,nu,om): return self.value( nu,om,1 )
  def si(self,nu,om): return self.value( nu,om,2 )

  def assign(self,nu,om,flavor,value):
    if flavor==0: self.hedin_ch[ nu+N_nu//2,om ] = value
    if flavor==1: self.hedin_sp[ nu+N_nu//2,om ] = value
    if flavor==2: self.hedin_si[ nu+N_nu//2,om ] = value

  def assign_ch(self,nu,om,value): return self.assign(nu,om,0,value)
  def assign_sp(self,nu,om,value): return self.assign(nu,om,1,value)
  def assign_si(self,nu,om,value): return self.assign(nu,om,2,value)

  def update(self,hd,flavor,xi,xi0):
    for nu in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        xi_ = xi
        if om==0: xi_ = xi0
        if flavor==0:
          value = xi_/hd.ch(nu,om) + (1.-xi_)/self.ch(nu,om)
          self.assign_ch( nu,om,1./value )
        elif flavor==1:
          value = xi_/hd.sp(nu,om) + (1.-xi_)/self.sp(nu,om)
          self.assign_sp( nu,om,1./value )
        elif flavor==2:
          value = xi_/hd.si(nu,om) + (1.-xi_)/self.si(nu,om)
          self.assign_si( nu,om,1./value )

  def update_ch(self,hd,xi,xi0):
    self.update(hd,0,xi,xi0)

  def update_sp(self,hd,xi,xi0):
    self.update(hd,1,xi,xi0)

  def update_si(self,hd,xi,xi0):
    self.update(hd,2,xi,xi0)


class vtx_container():
  def __init__(self):
    self.mb_ch = np.zeros( (N_om,N_nu,N_nu) , dtype=complex )
    self.mb_sp = np.zeros( (N_om,N_nu,N_nu) , dtype=complex )
    self.mb_si = np.zeros( (N_om,N_nu,N_nu) , dtype=complex )
    self.mb_tr = np.zeros( (N_om,N_nu,N_nu) , dtype=complex )

  def value(self,nu1,nu2,om,flavor):
    if om<0:
      return np.conjugate( self.value(-nu2-1,-nu1-1,-om,flavor) )
    else:
      if (nu1<nu_min) or (nu1>nu_max-1) or (nu2<nu_min) or (nu2>nu_max-1) or (om>om_max-1):
        return 0.0
      else:
        if flavor==0: return self.mb_ch[ om,nu1+N_nu//2,nu2+N_nu//2 ] 
        if flavor==1: return self.mb_sp[ om,nu1+N_nu//2,nu2+N_nu//2 ] 
        if flavor==2: return self.mb_si[ om,nu1+N_nu//2,nu2+N_nu//2 ] 
        if flavor==3: return self.mb_tr[ om,nu1+N_nu//2,nu2+N_nu//2 ] 
  
  def ch(self,nu1,nu2,om): return self.value( nu1,nu2,om,0 )
  def sp(self,nu1,nu2,om): return self.value( nu1,nu2,om,1 )
  def si(self,nu1,nu2,om): return self.value( nu1,nu2,om,2 )
  def tr(self,nu1,nu2,om): return self.value( nu1,nu2,om,3 )

  def assign(self,nu1,nu2,om,flavor,value):
    if flavor==0: self.mb_ch[ om,nu1+N_nu//2,nu2+N_nu//2 ] = value
    if flavor==1: self.mb_sp[ om,nu1+N_nu//2,nu2+N_nu//2 ] = value
    if flavor==2: self.mb_si[ om,nu1+N_nu//2,nu2+N_nu//2 ] = value
    if flavor==3: self.mb_tr[ om,nu1+N_nu//2,nu2+N_nu//2 ] = value

  def assign_ch(self,nu1,nu2,om,value): return self.assign(nu1,nu2,om,0,value)
  def assign_sp(self,nu1,nu2,om,value): return self.assign(nu1,nu2,om,1,value)
  def assign_si(self,nu1,nu2,om,value): return self.assign(nu1,nu2,om,2,value)
  def assign_tr(self,nu1,nu2,om,value): return self.assign(nu1,nu2,om,3,value)



class w_container():
  def __init__(self,g_,l_):
    self.g = g_
    self.l = l_
    self.pi_ch = np.zeros( N_om , dtype=complex )
    self.pi_sp = np.zeros( N_om , dtype=complex )
    self.pi_si = np.zeros( N_om , dtype=complex )

  def value(self,om,flavor):
    if om<0:
      return np.conjugate( self.value(-om,flavor) )
    else:
      if (om>om_max-1):
        if flavor == 0: return U_ch
        if flavor == 1: return U_sp
        if flavor == 2: return U_si
      else:
        if flavor == 0: return U_ch / (1. - self.pi_ch[ om ] * U_ch)
        if flavor == 1: return U_sp / (1. - self.pi_sp[ om ] * U_sp)
        if flavor == 2: return U_si / (1. - self.pi_si[ om ] * U_si / 2.) #regardless of definition, an ugly factor 2 appears somewhere in the pp channel

  def value_pi(self,om,flavor):
    if om<0:
      return np.conjugate( self.value(-om,flavor) )
    else:
      if (om>om_max-1):
        if flavor == 0: return 0.
        if flavor == 1: return 0.
        if flavor == 2: return 0.
      else:
        if flavor == 0: return self.pi_ch[ om ]
        if flavor == 1: return self.pi_sp[ om ]
        if flavor == 2: return self.pi_si[ om ]

  def assign_pi(self,om,flavor,value):
    if flavor==0: self.pi_ch[ om ] = value
    if flavor==1: self.pi_sp[ om ] = value
    if flavor==2: self.pi_si[ om ] = value

  def assign_pi_ch(self,om,value): return self.assign_pi(om,0,value)
  def assign_pi_sp(self,om,value): return self.assign_pi(om,1,value)
  def assign_pi_si(self,om,value): return self.assign_pi(om,2,value)

  def ch(self,om):
    return self.value(om,0)

  def sp(self,om):
    return self.value(om,1)

  def si(self,om):
    return self.value(om,2)

  def value_pi_ch(self,om):
    return self.value_pi(om,0)

  def value_pi_sp(self,om):
    return self.value_pi(om,1)

  def value_pi_si(self,om):
    return self.value_pi(om,2)

  def update_pi(self,xi):
    for om in range(om_min,om_max):
      result_ch = 0.0
      result_sp = 0.0
      result_si = 0.0
      result_si0 = 0.0
      result_bubble = 0.0
      result_pp_bubble = 0.0
      for nu in range(nu_min,nu_max):
        result_ch        += self.g.value(nu) * self.g.value(nu+om)   * (self.l.ch(nu,om) - 1.)
        result_sp        += self.g.value(nu) * self.g.value(nu+om)   * (self.l.sp(nu,om) - 1.)
        result_si        += self.g.value(nu) * self.g.value(om-nu-1) * (self.l.si(nu,om) + 1.)
        result_bubble    += self.g.value(nu) * self.g.value(nu+om)   - 1./( (1j*nu_m(nu) )*( 1j*nu_m(nu+om)   ) )
        result_pp_bubble += self.g.value(nu) * self.g.value(om-nu-1) - 1./( (1j*nu_m(nu) )*( 1j*nu_m(om-nu-1) ) )
        result_si0       += self.g.value(nu) * self.g.value(om-nu-1) * self.l.si(nu,om)

      value_ch = -beta*delta(om)/4. + result_bubble/beta + result_ch/beta
      self.pi_ch[ om ] = xi * value_ch + (1.-xi) * self.pi_ch[om]

      value_sp = -beta*delta(om)/4. + result_bubble/beta + result_sp/beta
      self.pi_sp[ om ] = xi * value_sp + (1.-xi) * self.pi_sp[om]

      if om==0:
        value_si = -beta*delta(om)/4. - result_pp_bubble/beta + result_si/beta
      else:
        value_si = -result_pp_bubble/beta + result_si/beta
      self.pi_si[ om ] = xi * value_si + (1.-xi) * self.pi_si[om]



class g_container():
  def __init__(self,g0_):
    self.sigma = np.zeros(nu_max,dtype=np.complex_)
    self.g0 = g0_
    self.l = hedin_container()
    self.w = w_container(self.g0,self.l)

  def set_pointers(self,w_,l_):
    self.w = w_
    self.l = l_

  def value_sigma(self,nu):
    if (nu<0):
      return np.conjugate( self.value_sigma(-nu-1) )
    else:
      return self.sigma[ nu ]

  def value(self,nu):
    if (nu < nu_max): return 1. / ( 1./self.g0(nu) - self.value_sigma(nu) )
    else:             return 1. / ( 1.j * nu_m(nu) )

  def assign_sigma(self,nu,value):
    self.sigma[nu] = value

  def update_sigma(self,xi):
    sigma = np.zeros(nu_max,dtype=np.complex_)
    for nu in range(0,nu_max):
      result = 0.0
      for om in range(0,om_max):
        result += self.value(nu+om) * ( self.w.ch(om) * self.l.ch(nu,om) + self.w.sp(om) * self.l.sp(nu,om) )
      sigma[nu] = U/2. - result/(2.*beta)
    for nu in range(0,nu_max):
      self.sigma[nu] = xi * sigma[nu] + (1.-xi) * self.sigma[nu]



class bse():
  def __init__(self,g_,krnl_,mb_):
    self.krnl=krnl_
    self.mb=mb_
    self.g = g_

  def update_mbe(self,xi):
    A = np.zeros( (N_nu,N_nu) , dtype=complex )
    for om in range(om_min,om_max):
      # charge
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          A[nu1-nu_min,nu2-nu_min] = self.krnl.ch(nu1,nu2,om) * self.g.value(nu2) * self.g.value(nu2+om) / beta
      evals, Umat = eigensystem.eig(A)
      Lambda = np.diag( evals / (1. - evals) )
      A = np.matmul(Umat,Lambda)
      Uinv = np.linalg.inv(Umat)
      Lambda = np.matmul(A,Uinv)
      Umat = np.zeros( (N_nu,N_nu) , dtype=complex )
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          Umat[nu1-nu_min,nu2-nu_min] = self.krnl.ch(nu1,nu2,om)
      A = np.matmul(Lambda,Umat)
      self.mb.mb_ch[om,:,:] = (1.-xi)*self.mb.mb_ch[om,:,:] + xi*A
      if check_bse:
        mm = minmax_real([A],nu_min,nu_max,om) 
        plot_vertex_real( A ,nu_min,nu_max,om,mm[0],mm[1],title='ch',filename='bse_ch')
      A = np.zeros( (N_nu,N_nu) , dtype=complex )
      print "BSE ch om: ", om

      # spin
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          A[nu1-nu_min,nu2-nu_min] = self.krnl.sp(nu1,nu2,om) * self.g.value(nu2) * self.g.value(nu2+om) / beta
      evals, Umat = eigensystem.eig(A)
      Lambda = np.diag( evals / (1. - evals) )
      A = np.matmul(Umat,Lambda)
      Uinv = np.linalg.inv(Umat)
      Lambda = np.matmul(A,Uinv)
      Umat = np.zeros( (N_nu,N_nu) , dtype=complex )
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          Umat[nu1-nu_min,nu2-nu_min] = self.krnl.sp(nu1,nu2,om)
      A = np.matmul(Lambda,Umat)
      self.mb.mb_sp[om,:,:] = (1.-xi)*self.mb.mb_sp[om,:,:] + xi*A
      if check_bse:
        mm = minmax_real([A],nu_min,nu_max,om) 
        plot_vertex_real( A ,nu_min,nu_max,om,mm[0],mm[1],title='sp',filename='bse_sp')
      A = np.zeros( (N_nu,N_nu) , dtype=complex )
      print "BSE sp om: ", om

      # singlet
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          A[nu1-nu_min,nu2-nu_min] = (-0.5) * self.krnl.si(nu1,nu2,om) * self.g.value(nu2) * self.g.value(om-nu2-1) / beta
      evals, Umat = eigensystem.eig(A)
      Lambda = np.diag( evals / (1. - evals) )
      A = np.matmul(Umat,Lambda)
      Uinv = np.linalg.inv(Umat)
      Lambda = np.matmul(A,Uinv)
      Umat = np.zeros( (N_nu,N_nu) , dtype=complex )
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          Umat[nu1-nu_min,nu2-nu_min] = self.krnl.si(nu1,nu2,om)
      A = np.matmul(Lambda,Umat)
      self.mb.mb_si[om,:,:] = (1.-xi)*self.mb.mb_si[om,:,:] + xi*A
      if check_bse:
        mm = minmax_real([A],nu_min,nu_max,om) 
        plot_vertex_real( A ,nu_min,nu_max,om,mm[0],mm[1],title='si',filename='bse_si')
      A = np.zeros( (N_nu,N_nu) , dtype=complex )
      print "BSE si om: ", om

      # triplet
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          A[nu1-nu_min,nu2-nu_min] = (+0.5) * self.krnl.tr(nu1,nu2,om) * self.g.value(nu2) * self.g.value(om-nu2-1) / beta
      evals, Umat = eigensystem.eig(A)
      Lambda = np.diag( evals / (1. - evals) )
      A = np.matmul(Umat,Lambda)
      Uinv = np.linalg.inv(Umat)
      Lambda = np.matmul(A,Uinv)
      Umat = np.zeros( (N_nu,N_nu) , dtype=complex )
      for nu1 in range(nu_min,nu_max):
        for nu2 in range(nu_min,nu_max):
          Umat[nu1-nu_min,nu2-nu_min] = self.krnl.tr(nu1,nu2,om)
      A = np.matmul(Lambda,Umat)
      self.mb.mb_tr[om,:,:] = (1.-xi)*self.mb.mb_tr[om,:,:] + xi*A
      if check_bse:
        mm = minmax_real([A],nu_min,nu_max,om) 
        plot_vertex_real( A ,nu_min,nu_max,om,mm[0],mm[1],title='tr',filename='bse_tr')
      A = np.zeros( (N_nu,N_nu) , dtype=complex )
      A = np.zeros( (N_nu,N_nu) , dtype=complex )
      print "BSE tr om: ", om


######################################################

#set up exact solution
w  = w()                  #screened interaction
hd = hedin()              #exact Hedin vertices
th = thunstroem()         #exact full vertex
nb = nabla(w,hd)          #exact U-reducible vertices
fi = threeleg_firr(th)    #exact fully gg-irreducible vertices
mb = mbe(th,nb)           #exact U-irreducible vertices


if (check_threeleg_firr):
  om_=0
  plt.plot([fi.ch(nu,om_).real     for nu in range(nu_min,nu_max)],color='r',marker='o')
  plt.show()
  
  plt.plot([fi.sp(nu,om_).real     for nu in range(nu_min,nu_max)],color='g',marker='o')
  plt.show()
  
  plt.plot([fi.si(nu,om_).real     for nu in range(nu_min,nu_max)],color='b',marker='o')
  plt.show()

##check relation between gamma and f
if(check_lambda_firr):
  lambda_firr = vtx_container()
  for nu1 in range(nu_min,nu_max):
    for nu2 in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        lambda_firr.assign_ch( nu1 , nu2 , om , th.firr_ch(nu1,nu2,om)-U_ch )
        lambda_firr.assign_sp( nu1 , nu2 , om , th.firr_sp(nu1,nu2,om)-U_sp )
        lambda_firr.assign_si( nu1 , nu2 , om , th.firr_si(nu1,nu2,om)-U_si )
        lambda_firr.assign_tr( nu1 , nu2 , om , th.firr_tr(nu1,nu2,om) )
  
  om_=0
  mm = minmax_real([lambda_firr.mb_ch[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( lambda_firr.mb_ch[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title='firr_ch',filename='firr_ch')
  
  mm = minmax_real([lambda_firr.mb_sp[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( lambda_firr.mb_sp[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title='firr_sp',filename='firr_sp')
  
  mm = minmax_real([lambda_firr.mb_si[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( lambda_firr.mb_si[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title='firr_si',filename='firr_si')
  
  mm = minmax_real([lambda_firr.mb_tr[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( lambda_firr.mb_tr[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title='firr_tr',filename='firr_tr')

if(check_phi):
  phi1 = vtx_container()
  for nu1 in range(nu_min,nu_max):
    for nu2 in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        phi1.assign_ch( nu1 , nu2 , om , th.phi_ch(nu1,nu2,om) )
        phi1.assign_sp( nu1 , nu2 , om , th.phi_sp(nu1,nu2,om) )
        phi1.assign_si( nu1 , nu2 , om , th.phi_si(nu1,nu2,om) )
        phi1.assign_tr( nu1 , nu2 , om , th.phi_tr(nu1,nu2,om) )
  
  om_=0
  #mm = minmax_real([phi1.mb_ch[om_,:,:],phi1.mb_sp[om_,:,:],phi1.mb_si[om_,:,:],phi1.mb_tr[om_,:,:]],nu_min,nu_max,om_) 
  mm = minmax_real([phi1.mb_ch[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( phi1.mb_ch[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$\Phi^{ph,\mathrm{ch}}$',filename='phi_ch')
  
  mm = minmax_real([phi1.mb_sp[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( phi1.mb_sp[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$\Phi^{ph,\mathrm{sp}}$',filename='phi_sp')
  
  mm = minmax_real([phi1.mb_si[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( phi1.mb_si[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$\Phi^{pp,\mathrm{si}}$',filename='phi_si')
  
  mm = minmax_real([phi1.mb_tr[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( phi1.mb_tr[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$\Phi^{pp,\mathrm{tr}}$',filename='phi_tr')

if(check_mbe):
  mbe1 = vtx_container()
  for nu1 in range(nu_min,nu_max):
    for nu2 in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        mbe1.assign_ch( nu1 , nu2 , om , mb.ch(nu1,nu2,om) )
        mbe1.assign_sp( nu1 , nu2 , om , mb.sp(nu1,nu2,om) )
        mbe1.assign_si( nu1 , nu2 , om , mb.si(nu1,nu2,om) )
        mbe1.assign_tr( nu1 , nu2 , om , mb.tr(nu1,nu2,om) )
  
  om_=0
  #mm = minmax_real([mbe1.mb_ch[om_,:,:],mbe1.mb_sp[om_,:,:],mbe1.mb_si[om_,:,:],mbe1.mb_tr[om_,:,:]],nu_min,nu_max,om_) 
  mm = minmax_real([mbe1.mb_ch[om_,:,:]],nu_min,nu_max,om_) 
  plot_vertex_real( mbe1.mb_ch[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$M^{ph,\mathrm{ch}}$',filename='mbe_ch')
                                                                                                                
  mm = minmax_real([mbe1.mb_sp[om_,:,:]],nu_min,nu_max,om_)                                                     
  plot_vertex_real( mbe1.mb_sp[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$M^{ph,\mathrm{sp}}$',filename='mbe_sp')
                                                                                                                
  mm = minmax_real([mbe1.mb_si[om_,:,:]],nu_min,nu_max,om_)
  #plot_vertex_real( mbe1.mb_si[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$M^{pp,\mathrm{si}}$',filename='mbe_si')
  plot_vertex_real( mbe1.mb_si[om_,:,:] ,nu_min,nu_max,om_,mm[0],-mm[0],title=r'$M^{pp,\mathrm{si}}$',filename='mbe_si')
  #for singlet channel automatic cbrange doesn't work well       ^   ^
                                                                                                                
  mm = minmax_real([mbe1.mb_tr[om_,:,:]],nu_min,nu_max,om_)                                                     
  plot_vertex_real( mbe1.mb_tr[om_,:,:] ,nu_min,nu_max,om_,mm[0],mm[1],title=r'$M^{pp,\mathrm{tr}}$',filename='mbe_tr')

if(check_gamma_consistency):
  gamma = vtx_container()
  fullv = vtx_container()
  bse_full = vtx_container()
  for nu1 in range(nu_min,nu_max):
    for nu2 in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        gamma.assign_ch( nu1 , nu2 , om , th.gamma(nu1,nu2,om,0) )
        gamma.assign_sp( nu1 , nu2 , om , th.gamma(nu1,nu2,om,1) )
        gamma.assign_si( nu1 , nu2 , om , th.gamma(nu1,nu2,om,2) )
        gamma.assign_tr( nu1 , nu2 , om , th.gamma(nu1,nu2,om,3) )
        fullv.assign_ch( nu1 , nu2 , om , th.ch(nu1,nu2,om) )
        fullv.assign_sp( nu1 , nu2 , om , th.sp(nu1,nu2,om) )
        fullv.assign_si( nu1 , nu2 , om , th.si(nu1,nu2,om) )
        fullv.assign_tr( nu1 , nu2 , om , th.tr(nu1,nu2,om) )
        bse_full.assign_ch( nu1 , nu2 , om , delta(nu1-nu2)+g(nu1)*g(nu1+om)*th.ch(nu1,nu2,om)/beta )
        bse_full.assign_sp( nu1 , nu2 , om , delta(nu1-nu2)+g(nu1)*g(nu1+om)*th.sp(nu1,nu2,om)/beta )
        bse_full.assign_si( nu1 , nu2 , om , delta(nu1-nu2)-g(nu1)*g(om-nu1-1)*th.si(nu1,nu2,om)/(2.*beta) )
        bse_full.assign_tr( nu1 , nu2 , om , delta(nu1-nu2)+g(nu1)*g(om-nu1-1)*th.tr(nu1,nu2,om)/(2.*beta) )
  
  om_=3
  bse_full_inv_ch=np.linalg.inv(bse_full.mb_ch[om_,:,:])
  bse_full_inv_sp=np.linalg.inv(bse_full.mb_sp[om_,:,:])
  bse_full_inv_si=np.linalg.inv(bse_full.mb_si[om_,:,:])
  bse_full_inv_tr=np.linalg.inv(bse_full.mb_tr[om_,:,:])
  
  gamma_ch = np.matmul(fullv.mb_ch[om_,:,:],bse_full_inv_ch)
  gamma_sp = np.matmul(fullv.mb_sp[om_,:,:],bse_full_inv_sp)
  gamma_si = np.matmul(fullv.mb_si[om_,:,:],bse_full_inv_si)
  gamma_tr = np.matmul(fullv.mb_tr[om_,:,:],bse_full_inv_tr)
  
  mm = minmax_real([gamma.mb_ch[om_,:,:]-gamma_ch],nu_min,nu_max,om_) 
  plot_vertex_real( gamma.mb_ch[om_,:,:]-gamma_ch ,nu_min,nu_max,om_,mm[0],mm[1],title='ch')
  
  mm = minmax_real([gamma.mb_sp[om_,:,:]-gamma_sp],nu_min,nu_max,om_) 
  plot_vertex_real( gamma.mb_sp[om_,:,:]-gamma_sp ,nu_min,nu_max,om_,mm[0],mm[1],title='sp')
  
  mm = minmax_real([gamma.mb_si[om_,:,:]-gamma_si],nu_min,nu_max,om_) 
  plot_vertex_real( gamma.mb_si[om_,:,:]-gamma_si ,nu_min,nu_max,om_,mm[0],mm[1],title='si')
  
  mm = minmax_real([gamma.mb_tr[om_,:,:]-gamma_tr],nu_min,nu_max,om_) 
  plot_vertex_real( gamma.mb_tr[om_,:,:]-gamma_tr ,nu_min,nu_max,om_,mm[0],mm[1],title='tr')
  
  print "ch: ", gamma.mb_ch[om_,:,:]-gamma_ch
  print "sp: ", gamma.mb_sp[om_,:,:]-gamma_sp
  print "si: ", gamma.mb_si[om_,:,:]-gamma_si
  print "tr: ", gamma.mb_tr[om_,:,:]-gamma_tr

#######################################################

#store fully irreducible vertex in container
fi_p3 = hedin_container()
for nu in range(nu_min,nu_max):
  for om in range(om_min,om_max):
    if parquet_approximation == True:
      fi_p3.assign_ch( nu , om , +1. ) 
      fi_p3.assign_sp( nu , om , +1. ) 
      fi_p3.assign_si( nu , om , -1. )
    else:
      fi_p3.assign_ch( nu , om , fi.ch(nu,om) ) 
      fi_p3.assign_sp( nu , om , fi.sp(nu,om) ) 
      fi_p3.assign_si( nu , om , fi.si(nu,om) )

firr_p3 = vtx_container()
for nu1 in range(nu_min,nu_max):
  for nu2 in range(nu_min,nu_max):
    for om in range(om_min,om_max):
      if parquet_approximation == True:
        firr_p3.assign_ch( nu1 , nu2 , om , 0. )#parquet approximation corresponds to Lambda^firr - U = 0 
        firr_p3.assign_sp( nu1 , nu2 , om , 0. )
        firr_p3.assign_si( nu1 , nu2 , om , 0. )
      else:
        firr_p3.assign_ch( nu1 , nu2 , om , th.firr_ch(nu1,nu2,om)-U_ch )#TODO: Implement the fully irr. vertex of AL
        firr_p3.assign_sp( nu1 , nu2 , om , th.firr_sp(nu1,nu2,om)-U_sp ) 
        firr_p3.assign_si( nu1 , nu2 , om , th.firr_si(nu1,nu2,om)-U_si )

#initial guess for the reducible Hedin vertices
beta -= delta_beta
hd0_p3 = hedin_container()
for nu in range(nu_min,nu_max):
  for om in range(om_min,om_max):
    if initial_guess == 0: #set Hedin vertices to +-1
      hd0_p3.assign_ch( nu , om , +1. ) 
      hd0_p3.assign_sp( nu , om , +1. ) 
      hd0_p3.assign_si( nu , om , -1. )
    elif initial_guess == 1: #set to fully irreducible vertex (may be worse than +-1)
      hd0_p3.assign_ch( nu , om , fi_p3.ch(nu,om) ) 
      hd0_p3.assign_sp( nu , om , fi_p3.sp(nu,om) ) 
      hd0_p3.assign_si( nu , om , fi_p3.si(nu,om) )
    elif initial_guess == 2: #'guess' the exact Hedin vertex 
      hd0_p3.assign_ch( nu , om , hd.ch(nu,om) ) 
      hd0_p3.assign_sp( nu , om , hd.sp(nu,om) ) 
      hd0_p3.assign_si( nu , om , hd.si(nu,om) )
beta += delta_beta

#initialize Green's function
#g_p3 = g_container(g0,w,hd)
g_p3 = g_container(g0) #g is now the non-interacting Green's function, to calculate a self-energy we still have to set w and lambda!
if initial_guess == 2:
  beta -= delta_beta
  for nu in range(0,nu_max):
    g_p3.assign_sigma(nu,sigma(nu))
  beta += delta_beta

#initialize screened interaction
w_p3 = w_container(g_p3,hd0_p3)
w_p3.update_pi(xi=1.)

#now that Hedin vertex hd0 and w have been initialized, we can set the pointers of g to these objects
g_p3.set_pointers(w_p3,hd0_p3)

#create a second instance to save intermediate results
hd1_p3 = hedin_container()

#create MBE container and functor
mbe_p3 = vtx_container()
mbe_functor_p3 = mbe_vtx(mbe_p3)

#initiate a U-reducible class with hd0_p3 (the 'current' result)
nb_p3 = nabla(w_p3,hd0_p3)    

#define kernel
kernel_p3 = kernel(nb_p3,mbe_functor_p3,firr_p3)

#define BSE
bse_functor_p3 = bse(g_p3,kernel_p3,mbe_p3)

#initiate P3 equations with hd0_p3, notice that fi_p3 remains fixed through the iterations
p3 = parquet3(g_p3,nb_p3,mbe_functor_p3,fi_p3)

it1_max=30
it2_max=3
it_warmup=5

xi = 0.5
xi_w = 0.5
xi_sigma = 0.5
xi_bse = 0.5

xi_ch  = xi # 0.05
xi_sp  = xi # 1.
xi_si  = xi # 0.05

xi_ch0  = xi
xi_sp0  = xi
xi_si0  = xi

# main loop
for it1 in range(it1_max):
  if it1 == it_warmup: xi_sp0 = xi
  for om1 in range(om_min_,om_max_,om_inc):
    plt.title(str(-1))
    plt.subplot(421)
    plt.plot([hd.ch(nu,om1).real     for nu in range(nu_min,nu_max)],color='r',marker='o',linestyle='')
    plt.plot([hd0_p3.ch(nu,om1).real for nu in range(nu_min,nu_max)],color='r')
    
    plt.subplot(423)
    plt.plot([hd.sp(nu,om1).real     for nu in range(nu_min,nu_max)],color='g',marker='o',linestyle='')
    plt.plot([hd0_p3.sp(nu,om1).real for nu in range(nu_min,nu_max)],color='g')
    
    plt.subplot(425)
    plt.plot([hd.si(nu,om1).real     for nu in range(nu_min,nu_max)],color='b',marker='o',linestyle='')
    plt.plot([hd0_p3.si(nu,om1).real for nu in range(nu_min,nu_max)],color='b')

    plt.subplot(422)
    plt.plot([pi_ch(om).real for om in range(om_min,om_max)],color='r',marker='o',linestyle='')
    plt.plot([w_p3.value_pi_ch(om).real for om in range(om_min,om_max)],color='r')

    plt.subplot(424)
    plt.plot([pi_sp(om).real for om in range(om_min,om_max)],color='g',marker='o',linestyle='')
    plt.plot([w_p3.value_pi_sp(om).real for om in range(om_min,om_max)],color='g')

    plt.subplot(426)
    plt.plot([pi_si(om).real for om in range(om_min,om_max)],color='b',marker='o',linestyle='')
    plt.plot([w_p3.value_pi_si(om).real for om in range(om_min,om_max)],color='b')

#    g_p3.update_sigma()
    plt.subplot(427)
    plt.plot([           sigma(nu).real for nu in range(0,nu_max)],color='g',marker='o',linestyle='')
    plt.plot([           sigma(nu).imag for nu in range(0,nu_max)],color='r',marker='o',linestyle='')
    plt.plot([g_p3.value_sigma(nu).real for nu in range(0,nu_max)],color='g')
    plt.plot([g_p3.value_sigma(nu).imag for nu in range(0,nu_max)],color='r')

    plt.subplot(428)
    plt.plot([   nb.ch(nu,0,0).real for nu in range(nu_min,nu_max)],color='r',marker='o',linestyle='')
    plt.plot([nb_p3.ch(nu,0,0).real for nu in range(nu_min,nu_max)],color='r')
    plt.plot([   nb.sp(nu,0,0).real for nu in range(nu_min,nu_max)],color='g',marker='o',linestyle='')
    plt.plot([nb_p3.sp(nu,0,0).real for nu in range(nu_min,nu_max)],color='g')
    plt.plot([   nb.si(nu,0,0).real for nu in range(nu_min,nu_max)],color='b',marker='o',linestyle='')
    plt.plot([nb_p3.si(nu,0,0).real for nu in range(nu_min,nu_max)],color='b')
  plt.pause(plot_pause)

  bse_functor_p3.update_mbe(xi_bse)

  #charge
  for it2 in range(it2_max):
    for nu in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        hd1_p3.assign_ch( nu,om,p3.ch(nu,om) )
    #update hd0 with hd1
    hd0_p3.update_ch(hd1_p3,xi_ch,xi_ch0)
    for om1 in range(om_min_,om_max_,om_inc):
      plt.subplot(421)
      plt.title(str(it1)+' '+str(it2))
      plt.plot([hd0_p3.ch(nu,om1).real for nu in range(nu_min,nu_max)],color='r')
      plt.pause(plot_pause)

  #spin
  for it2 in range(it2_max):
    for nu in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        hd1_p3.assign_sp( nu,om,p3.sp(nu,om) )
    #update hd0 with hd1
    hd0_p3.update_sp(hd1_p3,xi_sp,xi_sp0)
    for om1 in range(om_min_,om_max_,om_inc):
      plt.subplot(423)
      plt.title(str(it1)+' '+str(it2))
      plt.plot([hd0_p3.sp(nu,om1).real for nu in range(nu_min,nu_max)],color='g')
      plt.pause(plot_pause)
  
  #singlet
  for it2 in range(it2_max):
    for nu in range(nu_min,nu_max):
      for om in range(om_min,om_max):
        hd1_p3.assign_si( nu,om,p3.si(nu,om) )
    #update hd0 with hd1
    hd0_p3.update_si(hd1_p3,xi_si,xi_si0)
    for om1 in range(om_min_,om_max_,om_inc):
      plt.subplot(425)
      plt.title(str(it1)+' '+str(it2))
      plt.plot([hd0_p3.si(nu,om1).real for nu in range(nu_min,nu_max)],color='b')
      plt.pause(plot_pause)

  for it2 in range(it2_max):
    w_p3.update_pi(xi_w)
    g_p3.update_sigma(xi_sigma)

    plt.subplot(422)
    plt.plot([w_p3.value_pi_ch(om).real for om in range(om_min,om_max)],color='r')
    
    plt.subplot(424)
    plt.plot([w_p3.value_pi_sp(om).real for om in range(om_min,om_max)],color='g')
    
    plt.subplot(426)
    plt.plot([w_p3.value_pi_si(om).real for om in range(om_min,om_max)],color='b')

  plt.pause(plot_pause)

  plt.subplot(427)
  plt.plot([           sigma(nu).real for nu in range(0,nu_max)],color='g',marker='o',linestyle='')
  plt.plot([           sigma(nu).imag for nu in range(0,nu_max)],color='r',marker='o',linestyle='')
  plt.plot([g_p3.value_sigma(nu).real for nu in range(0,nu_max)],color='g')
  plt.plot([g_p3.value_sigma(nu).imag for nu in range(0,nu_max)],color='r')
  plt.pause(plot_pause)
  
  plt.clf()

for om1 in range(om_min_,om_max_,om_inc):
  plt.subplot(421)
  plt.title("final")
  plt.plot([hd.ch(nu,om1).real     for nu in range(nu_min,nu_max)],color='r',marker='o')
  plt.plot([hd0_p3.ch(nu,om1).real for nu in range(nu_min,nu_max)],color='r')
  
  plt.subplot(423)
  plt.plot([hd.sp(nu,om1).real     for nu in range(nu_min,nu_max)],color='g',marker='o')
  plt.plot([hd0_p3.sp(nu,om1).real for nu in range(nu_min,nu_max)],color='g')
  
  plt.subplot(425)
  plt.plot([hd.si(nu,om1).real     for nu in range(nu_min,nu_max)],color='b',marker='o')
  plt.plot([hd0_p3.si(nu,om1).real for nu in range(nu_min,nu_max)],color='b')

  plt.subplot(422)
  plt.plot([pi_ch(om).real for om in range(om_min,om_max)],color='r',marker='o',linestyle='')
  plt.plot([w_p3.value_pi_ch(om).real for om in range(om_min,om_max)],color='r')

  plt.subplot(424)
  plt.plot([pi_sp(om).real for om in range(om_min,om_max)],color='g',marker='o',linestyle='')
  plt.plot([w_p3.value_pi_sp(om).real for om in range(om_min,om_max)],color='g')

  plt.subplot(426)
  plt.plot([pi_si(om).real for om in range(om_min,om_max)],color='b',marker='o',linestyle='')
  plt.plot([w_p3.value_pi_si(om).real for om in range(om_min,om_max)],color='b')

  plt.subplot(427)
  plt.plot([           sigma(nu).real for nu in range(0,nu_max)],color='g',marker='o',linestyle='')
  plt.plot([           sigma(nu).imag for nu in range(0,nu_max)],color='r',marker='o',linestyle='')
  plt.plot([g_p3.value_sigma(nu).real for nu in range(0,nu_max)],color='g')
  plt.plot([g_p3.value_sigma(nu).imag for nu in range(0,nu_max)],color='r')
  plt.pause(plot_pause)

  plt.subplot(428)
  plt.plot([   nb.ch(nu,0,0).real for nu in range(nu_min,nu_max)],color='r',marker='o',linestyle='')
  plt.plot([nb_p3.ch(nu,0,0).real for nu in range(nu_min,nu_max)],color='r')
  plt.plot([   nb.sp(nu,0,0).real for nu in range(nu_min,nu_max)],color='g',marker='o',linestyle='')
  plt.plot([nb_p3.sp(nu,0,0).real for nu in range(nu_min,nu_max)],color='g')
  plt.plot([   nb.si(nu,0,0).real for nu in range(nu_min,nu_max)],color='b',marker='o',linestyle='')
  plt.plot([nb_p3.si(nu,0,0).real for nu in range(nu_min,nu_max)],color='b')
plt.show()


