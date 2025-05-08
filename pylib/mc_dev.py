from utilities.ipynb_docgen import show,capture_hide
from pylib.data_setup import (set_theme, show_date, show_link)
from wtlike import simulation, WtLike, Timer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

set_theme(['dark'])
show(f"""# Monte Carlo BB light curve study 
""", id='top')
show_date()

sim_design_thoughts= f"""---
    ## Simulation procedure step
    We want to focus on a single step in the signal
    function. Rather than run BB for long periods, we
    isolate a handful of cells involved.

    ### Setup
    Take a set of data cells, make three groups:
    1. Before -- well before the measured transition. Combine these to one
    2. Bracket -- inbetween, keep to adjust
    3. After -- well after, combine as well

    ### Simulation
    1. Choose a step position within the bracket, default the measured one
    2. For each of the bracketed cells, create a new cell with the expected flux 
    3. Run BB with the new set

    ### Code requirements
    1. cell simulator<br>
    A cell is defined by a set of weights $w$ and $S$, the expected number of weights. ($B$ is zero,
    no background variation.)
    Our randomization means, for an expected flux ratio $f$, selecting a new 
    set of weights such that
    * The number of weights $n$ is poisson-selected such that $<n> = f S$.
    * The weights are chosen from the experimental distribution. `wtlike.simulation._Sampler`
    designed for this
    2. Interface to BB might need tweaking--but it is only an ordered set of cells in which it 
    finds partitions.
    """
# Test statistic for variability 
def TSvar(fitvals):
    """ fitvals: list of log-likelihood functions, one for each cell
    Returns the test statistic for the variability of the light curve.
    """
    return 0 if len(fitvals<2) else -2 * np.sum([f(1) for f in fitvals])

TS = lambda df : -2 * np.sum([f(1) for f in df.fit.values]) if len(df)>1 else 0

def generate_trials(wtl, ntrials, interval=None):
    with capture_hide():
        wtx = wtl.view(interval) if interval is not None else wtl
    interval = wtx.time_bins[2]
    tstart, tstop = wtl.start, wtl.stop

    def random_lc(reverse=False):
        with capture_hide():
            wsim = wtx.view(apply=simcell)
            return wsim.bb_view(reverse=reverse).fits
    
    show(f"""### Run {ntrials} trials, interval={interval} """)
    with Timer() as et:
        lcs = [random_lc() for i in range(ntrials)]
    show(f'{et}')
    return lcs

class Sampler(simulation._Sampler):
    
    def __init__(self, wdata,  **kwargs):
        """
        """
        nbins = kwargs.pop('nbins', 4000)
        y, _ = np.histogram(wdata,  np.linspace(0,1,nbins+1),  )
        super().__init__( y,  **kwargs)
        
    def uniform(self, size, alpha=0):
        return self._evaluate(np.linspace(0,1, size), alpha=alpha)
    
    def random(self, size, alpha=0):
        return self(size, alpha)

    @classmethod
    def test(cls, wdata, make_plot=False, **kwargs):
        alpha = kwargs.pop('alpha',0)   
        smp =   cls(wdata, **kwargs)  
        wunif = smp.uniform(len(wdata), alpha=alpha)
        # wran  = smp.random(len(wdata), alpha=alpha)
        if not make_plot: 
            stats = lambda w: f'{len(w)}|{np.mean(w):.2e}| {np.mean(w**2):.2e}'
            return f"""           
            | Sample  | N | $<w>$ | $<w^2>$ 
            | --------|--:|------:|--------:
            |data     | {stats(wdata)}
            |uniform  | {stats(wunif)}
            |random 1 | {stats(smp.random(len(wdata), alpha=alpha))}
            |random 2 | {stats(smp.random(len(wdata), alpha=alpha))}
            """
       
        bins = np.logspace(-5,0,101)  
        quality = lambda w: np.sum(w)/np.sqrt(np.sum(w**2))
        fig, (ax1,ax2,) =plt.subplots(nrows=2,figsize=(8,6), sharex=True, sharey=True)
        ax1.set(xscale='log', yscale='log', title='Data')
        ax1.hist(wdata, bins=bins, histtype='step',
                 label=f'mean {np.mean(wdata):.2e}\nquality {quality(wdata):.0f}');
        ax1.legend();    
        
        ax2.hist(wunif, bins=bins, histtype='step', 
                 label=f'mean {np.mean(wsim):.2e}\nquality {quality(wsim):.0f}')
        ax2.set(title='Uniform sampling'),ax2.legend();
        return fig
    

def test_sampler(wtl, make_plot=False):
    show(f"""## Exercise the weight sampler """)
    show(Sampler.test(wtl.photons.weight, make_plot=make_plot ));

def simulate(cell, weight_sampler, t=None, tw=None, in_place=True):
    """ Return a simulated version of the cell with a new set of weights.
    Optionally set the time and width
    Replace if `in_place` is set, otherwise return a copy."""
    simcell = cell if in_place else cell.copy() 
    simcell.w = weight_sampler.random(size=cell.S+cell.B)
    simcell.n = len(simcell.w)
    if t is not None: simcell.t=t
    if tw is not None: simcell.tw=tw
    return None if in_place else simcell

def check_lightcurve(wtl):
    
    show(f"""## Regenerate the light curve
    This source, {wtl.source_name}, was selected as having a single up-step in the original batch process.
    Run a new BB, forward and reversed""")
    with capture_hide('BB processing output') as output:
        bb, bbr = wtl.bb_view(), wtl.bb_view(reverse=True)
    show(output)
    
    plot_kw= dict(colors = 'none none blue'.split(), yscale='log', ylim=(0.2,5), source_name='')
    sns.set_context('notebook')
    fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(8,4), sharex=True, sharey=True,
                                 gridspec_kw=dict(hspace=0.1))
    bb.plot(ax=ax1, xlabel='', **plot_kw);
    bbr.plot(ax=ax2, **plot_kw)
    show(fig, caption=f"""BB-generated light curves for {wtl.source_name} with weekly binning. 
    The upper plot has BB algorithm running forward in time, the lower reversed.""")

def cell_simulation_test(wtl, interval):

    show(f""" ## Simulate cells
    """)
    
    from wtlike.loglike import LogLike
    # interval= 2*365
    with capture_hide():
        v = wtl.view(interval)
    show(f"""Get first {interval}-day cell from source {wtl.source_name}, create `LogLike` object""")
    cell = v.cells.iloc[0]
    ll = LogLike(cell)
    show(f"""LogLike object {str(ll)}
    <br>Its likelhood function:
    """)
    fig, ax=plt.subplots(figsize=(3,2))
    ll.plot(ax=ax)
    show(fig)
    
    mu = ll.S+ll.B
    
    show(f"""Now replace the weights<br>
    The number to generate, with flux=1, is {mu:.1f}
    """)
    wdata = wtl.photons.weight.values
    sampler = Sampler(wdata)

        
    n = 1000
    show(f"""* Make and fit {n} simulated copies""")
    simcells = [LogLike(simulate(cell,sampler, t=t)) for t in np.arange(0.5,n,1)]
    
    show(f"""* Plot a few...""")
    
    fig, axx = plt.subplots(ncols=4, nrows=2, sharex=True, sharey=True, 
                            figsize=(12,4))
    for ax, cell in zip(axx.flatten(),simcells):
        cell.plot(ax=ax)
    show(fig)
    
    ff = np.array([ simcell.fit_info()['flux'] for simcell in simcells])
    original_fit = ll.fit_info()
    show(f"""Check that mean and std are consistent with original fit:
    | | mean | sigma
    |--|--:|--:
    | original | {original_fit['flux']:.2f} | {original_fit['sig_flux']:.2f}
    | {n} simulations | {ff.mean():.2f} | {ff.std():.2f} """)


class SimCell:
    def __init__(self, wtl):
        self.sampler = Sampler( wtl.photons.weight.values)
        self.source_name = wtl.source_name

    def __call__(self, cell):
        if len(cell.w)<3:
            return cell # skip minimal data?
        n = cell.n = int(cell.S+cell.B)
        w = self.sampler.random(size=n)
        before = np.sum(cell.w)
        after = np.sum(w)
        assert before>0 and before!=after, f'{before:.3e} != {after:.3e}'
        cell.w = w
        return cell
    def __str__(self):
        return f'SimCell using weights from {self.source_name}'

TS = lambda df : -2 * np.sum([f(1) for f in df.fit.values])

def generate_trials(wtl, ntrials, interval=None):
    with capture_hide():
        wtx = wtl.view(interval) if interval is not None else wtl
    interval = wtx.time_bins[2]
    tstart, tstop = wtl.start, wtl.stop
    simcell = SimCell(wtl)

    def random_lc(reverse=False):
        with capture_hide():
            wsim = wtx.view(apply=simcell)
            return wsim.bb_view(reverse=reverse).fits
    
    show(f"""### Run {ntrials} trials, interval={interval} """)
    with Timer() as et:
        lcs = [random_lc() for i in range(ntrials)]
    show(f'{et}')
    return lcs
