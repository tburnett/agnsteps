from utilities.ipynb_docgen import show,capture_hide
from pylib.data_setup import (set_theme, show_date, show_link)
from wtlike import simulation, WtLike, Timer
from wtlike.loglike import LogLike, PoissonRep
from wtlike.lightcurve import fit_cells
from wtlike.bayesian import LikelihoodFitness

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

set_theme(['dark'])
show(f"""# Monte Carlo BB light curve study 
""", id='top')
show_date()

sim_design_thoughts= r"""---
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
    A cell is defined by a set of weights $w$ and $S$, the expected number of weights. ($\beta$ is zero,
    no background variation.)
    Our randomization means, for an expected flux ratio $f$, selecting a new 
    set of weights such that
    * The number of weights $n$ is poisson-selected such that $<n> = f S$.
    * The weights are chosen from the experimental distribution. `CellSim` is
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

def check_step(bb, showit=True):
    ff = bb.fits
    showq = show if showit else lambda x: None
    showq(f"""TSvar = {TS(ff):.1f} for {len(ff)} blocks""")
    if len(ff)==2:
        showq(f"""Single step observed at MJD {(tstep:=ff.t[0]+ff.tw[0]/2)}, 
             ratio {(ratio:=ff.fit[1].flux/ff.fit[0].flux):.2f}""")
        return tstep, round(TS(ff),1), ratio, 2
    else:
        showq(f"""No single step observed, {len(ff)} blocks""")
        return np.nan, round(TS(ff),1), np.nan, len(ff)
    


def check_lightcurve(wtl):
    
    show(f"""## Regenerate the light curve
    This source, {wtl.source_name}, was selected as having a single up-step in the original batch process.
    Run a new BB, forward and reversed""")
    with capture_hide('BB processing output') as output:
        bb, bbr = wtl.bb_view(), wtl.bb_view(reverse=True)
    show(output)
    
    plot_kw= dict(colors = 'none none lightgreen'.split(), yscale='log', ylim=(0.2,5), source_name='')
    sns.set_context('notebook')
    fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(8,4), sharex=True, sharey=True,
                                 gridspec_kw=dict(hspace=0.1))
    bb.plot(ax=ax1, xlabel='', **plot_kw);
    bbr.plot(ax=ax2, **plot_kw)
    show(fig, caption=f"""BB-generated light curves for {wtl.source_name} with weekly binning. 
    The upper plot has BB algorithm running forward in time, the lower reversed.""")

class BBsim:
    """ A class to simulate a BB light curve with a step"""
    def __init__(self, pv, step_time=57196):
    
        """pv: a WtLike object with cells
        step_time: time of the step, default 57196  
        """

        show(f"""## Setup BB fit sims with adjusted week cells
            Assume step at {step_time} """)
        self.pv = pv
        self.step_time = step_time
        self.cells = pv.cells
        first = pv.cells.iloc[0]
        last  = pv.cells.iloc[-1]
        self.before_sim = CellSim(first)
        self.after_sim = CellSim(last)

    def sim_view(self, step_time=None, seed=None ): #new_edges):
        """ Return a simulated view 
    
        """
        from wtlike.lightcurve import fit_cells
        if seed is not None: np.random.seed(seed)
    
        # basic copy of the WtLike object
        r = self.pv.view()
        incells = self.pv.cells

        # make a new set of cells by modifying the present set
        if step_time is None: step_time = self.step_time
        sim_cells = []
        for i,cell in incells.iloc[1:-1].iterrows():
            sim_cells.append( 
                (self.before_sim if cell.t<step_time else self.after_sim) (cell) )
            
        r.cells = pd.DataFrame([incells.iloc[0]]+sim_cells+
                                [incells.iloc[-1]], columns=incells.columns)

            
        # then add poisson fits
        with capture_hide():
            r.fits = fit_cells(self.pv.config, r.cells, )
        return r
        
    def runit(self, step_time=None):
        """Run a simulation
        """       
        with capture_hide():
            bbsimview = self.sim_view(step_time)
            ret = check_step(bbsimview, showit=False)
        return ret



def p_view(self, t1, t2 ): #new_edges):
    """ Return a WtLike view with arbitrary partition
    (here keep weeks in range t1 to t2, combining before and after )
    """
    from wtlike.cell_data import partition_cells
    from wtlike.lightcurve import fit_cells
    cells = self.cells
    edges = np.append(cells.t-cells.tw/2, cells.iloc[-1].t+cells.iloc[-1].tw/2)
    k,m = np.searchsorted(self.cells.t, [t1, t2])
    new_edges = np.append(
                    np.append(edges[0], edges[k:m+1]),   
                    edges[-1]);

    # basic copy
    r = self.view()

    # make new set of cells and add poisson fits
    r.cells = partition_cells(self.config, self.cells, new_edges)
    r.fits = fit_cells(self.config, r.cells, )
    return r


class CellSim:
    """ A class to simulate a cell with a new set of weights
    """

    def __init__(self, incell, rgen = None):
        """incell: a cell object for calibration
        """
        w = incell.w
        self.S, self.B = incell.S, incell.B
        self.N = len(w)
        self.rgen = rgen

        self.poiss_fit = PoissonRep(LogLike(incell))
        self.flux = self.poiss_fit.flux
                
        # the CDF of the weights
        cdf = stats.ecdf(w).cdf

        delta =0.01  # how far past sample edge to plot
        q = cdf.quantiles
        self.q = np.array( list(q) + [q[-1] + delta])
        self.yq = cdf.evaluate(q)

    

    def __repr__(self):
        return f"CellSim: S={self.S:.0f}, B={self.B:.0f}, N={self.N}, poiss={self.poiss_fit}" 
    
    def __call__(self, cell, inplace=False):
        """ Return a simulated version of the cell with a new set of weights.
        Flux corresponds to the input model
        """
        simcell = cell.copy() if not inplace else cell
        # poisson sample the number of weights, with mean = flux*cell.S+cell.B 
        # expected number, using flux from the setup cell. Use with Poisson
        with pd.option_context('mode.chained_assignment', None): # to ignore warning
            mu = self.flux*cell.S+cell.B 
            simcell.w = self._weight_gen(
                stats.uniform.rvs(
                    size=stats.poisson.rvs(mu, random_state=self.rgen),
                    random_state=self.rgen))
            simcell.n = len(simcell.w)
        return simcell    
        
    def _weight_gen(self, rgen):
        """Return sampled weight distribution 
        """
        return  self.q[np.searchsorted(self.yq, rgen)]
        
    def alpha_estimate(self, w):
        """ Return alpha and sigma_alpha derived from the weights $w$ and with the cell's
        signal and background estimates
        using the weak-signal approximation to the likelihood
        """  
        W, U  = np.sum(w), np.sum(w*w)
        return (W-self.S)/U, np.sqrt(1/U)
        
    @classmethod
    def check_weights(cls, cell, N=None):
        """ Test weight generation
        """
        sf = cls(cell)
        if N is  None: N = len(cell.w) 
        wunif = sf._weight_gen(np.linspace(0,1,N))
        def stat(w):
            alf = sf.alpha_estimate(w)
            return  f"{len(w):,}|{np.mean(w):.2e}| {np.mean(w**2):.2e}"\
                f"|{alf[0]:.3f} | {alf[1]:.3f}"
        return rf"""           
            | Sample  | N | $<w>$ | $<w^2>$ | $\alpha$ | $\sigma_\alpha$
            | --------|--:|------:|--------:|---------:|-----------
            |data     | {stat(cell.w)}
            |uniform  | {stat(wunif)}
            |random   | {stat(sf._weight_gen(stats.uniform.rvs( size=N)))}

            """
    @classmethod
    def plot_cdf(cls, cell, ax=None, **kwargs):
        """ Plot the CDF of the weights
        """
        sf = cls(cell)
        fig, ax = plt.subplots(figsize=(5, 4)) if ax is None else (ax.figure, ax)
        ax.plot(sf.q[:-1], sf.yq, label='CDF')
        ax.set(xlabel='Weight', ylabel='CDF', title='CDF of Weights',
               xscale='log',xlim=(None,1), ylim=(0,1), **kwargs)
        ax.legend()
        return fig    

        
