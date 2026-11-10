import unittest
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import expit
from workflows.tsba.validation.p0_oracle import source_terms,source_quadrature,analyze
from workflows.tsba.validation.static_score import schedules,new_model
from tsba.config import Config
from tsba.score import denoising_loss
import copy

class P0Tests(unittest.TestCase):
    def test_source_derivatives_and_quadrature(self):
        x,w=source_quadrature()
        for b in (0.,.25,.5):
            ce,g,f=source_terms(b,x,w)
            bt=torch.tensor(b,dtype=torch.float64,requires_grad=True)
            xt=torch.tensor(x); wt=torch.tensor(w); z=8*(xt-bt)
            loss=(wt*(torch.logaddexp(torch.zeros_like(z),z)-torch.sigmoid(8*(xt-.5))*z)).sum()
            grad=torch.autograd.grad(loss,bt,create_graph=True)[0]
            h=torch.autograd.grad(grad,bt)[0]
            self.assertAlmostEqual(g,float(grad),places=12);self.assertAlmostEqual(f,float(h),places=12)
            def fn(t):
                density=sum(np.exp(-.5*((t-m)/.5)**2)/(.5*np.sqrt(2*np.pi))/2 for m in (-.5,1.5))
                return density*(np.logaddexp(0,8*(t-b))-expit(8*(t-.5))*8*(t-b))
            exact=quad(fn,-8,9,epsabs=1e-12)[0]
            self.assertAlmostEqual(ce,exact,places=10)
    def test_schedule_and_adam_resume(self):
        cfg=Config(); short=schedules(2,0,updates=4);long=schedules(2,0,updates=16)
        for n in short['ids']:torch.testing.assert_close(short['ids'][n],long['ids'][n][:4],rtol=0,atol=0)
        target=torch.randn(8192,2,generator=torch.Generator().manual_seed(123),dtype=torch.float64)
        m=new_model(2,cfg);o=torch.optim.Adam(m.parameters(),lr=cfg.score_lr)
        def advance(m,o,start,end):
            for i in range(start,end):
                o.zero_grad();denoising_loss(m,target[long['ids'][8192][i]],.05,long['noise'][i]).backward();o.step()
        advance(m,o,0,4);state=copy.deepcopy(m.state_dict());opt=copy.deepcopy(o.state_dict());advance(m,o,4,16)
        r=new_model(2,cfg);r.load_state_dict(state);ro=torch.optim.Adam(r.parameters(),lr=cfg.score_lr);ro.load_state_dict(opt);advance(r,ro,4,16)
        for a,b in zip(m.parameters(),r.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_oracle_equilibrium(self):
        x,_=source_quadrature(256);r,_,_=analyze(1.1,.055,Config(),x)
        for k in ('population_source','finite512_posterior'):
            a=r[k];self.assertLess(abs(a['equilibrium_residual']),1e-12)
            self.assertLess(abs(a['trajectory']['final']['b']-a['equilibrium_b']),1e-8)
            self.assertGreater(a['equilibrium_ba'],r['reference']['initial_ba'])
        self.assertLess(abs(r['density_only']['final']['b']),1e-6)
