import numpy as np
import pandas as pd
from scipy import stats
from wtlike.loglike import LogLike, PoissonRep, GaussianRep

class CDFinverter:
    
    """ A class to invert the cumulative distribution function (CDF) of a set of weights.

    This allows sampling weights according to their cumulative distribution.
    """
    
    def __init__(self, w):
        """ Initialize the CDFinverter with a set of weights.

        Parameters:
        w : array-like
            The weights for which to invert the cumulative distribution function.
        """
        from scipy import stats

        cdf = stats.ecdf(w).cdf
        q = cdf.quantiles
        self.q = np.array( list(q), dtype=np.float32)
        self.yq = cdf.evaluate(q)

    def __call__(self, probs):
        """ Sample weights according to the cumulative distribution.

        Parameters:
        probs : array-like
            Probabilities at which to sample the weights.

        Returns:
        array-like
            Sampled weights corresponding to the input probabilities.
        """
        return self.q[np.searchsorted(self.yq, probs)].astype(np.float32)

    def plot(self, *, ax=None, label=None, **kwargs):
        """ Plot the CDF of the weights
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4)) if ax is None else (ax.figure, ax)
        ax.plot(self.q, self.yq, label=label )
        ax.set(xlabel='Weight', ylabel=label , title=kwargs.get('title', 'CDF of Weights'),
               xscale='log',xlim=(None,1), ylim=(0,1), **kwargs)
        return fig   

    def multi_plot(self):
        """Plots of the cumulative weight distribution and scatter 
        and scatter plot ofeight counts.
        """
        import matplotlib.pyplot as plt
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(10, 5), 
                        gridspec_kw={'width_ratios': [1, 1.5], "wspace": 0.5})
        self.plot(ax=ax1, )
        ax2.scatter(self.q, np.diff(np.insert(self.yq * len(self.q), 0, 0)), marker='o', s=5)
        ax2.set(xscale='log', xlabel='Weight values', ylabel='Counts', yscale='log')
        return fig


class Cell(pd.Series):
    """ A class representing a cell with weights.
    It is initialized with a "template" cell that provides expected flux and weight distribution.
    """

    def __init__(self, template_cell, *, name: str='', **kwargs):
        """ template_cell: Expect the input cell to be dict-like
            name: The name to assign to the cell.
        """
        super().__init__(template_cell, **kwargs)
        if name: self.name = name

        self.weight_sampler = CDFinverter(self.w)

        # characerize the template cell by fitting a Gaussian to its flux likelihood distribution
        fit= self.gaussian_fit().fit
        self['flux'] = fit['flux']
        self['sig_flux'] = fit['sig_flux']

    def __repr__(self):
        return f"""Cell name="{self.name}" flux={self.flux:.3f} +/- {self.sig_flux:.3f}, {len(self.w)} weights"""

    def poisson_fit(self):
        """ Fit a Poisson distribution to the cell's weights and return a PoissonRep object."""
        return PoissonRep(LogLike(self))

    def gaussian_fit(self):
        """ Fit a Gaussian distribution to the cell's weights
        
        Returns a GaussianRep object that can be used to estimate the flux and its uncertainty.
            need .fit['flux'] and .fit['sigma_flux'] to get the flux and its uncertainty
        """
        return GaussianRep(LogLike(self))

    # def get_cumulative_w(self, nsamples=400):
    #     """ Return a (bin edges, cumulative distribution) tuple for the cell's weights.
    #     """
    #     w_edges = np.logspace(-4, 0, nsamples)
    #     w_hist,_  = np.histogram(self.w, w_edges)
    #     Fw = np.insert(np.cumsum(w_hist), 0, 0)
    #     return w_edges, Fw/Fw[-1]

    # def weight_sampler(self, alpha=None):
    #     """ Return a function that can be used to sample weights from the cell's cumulative distribution."""
    #     if alpha is not None:
    #         raise NotImplementedError("alpha not used yet")
    #     w_edges, Fw = self.get_cumulative_w()
    #     def sampler(probs):
    #         return np.interp( probs, Fw, w_edges).astype(np.float32)            
    #     return sampler

    def expected_n(self, alpha=None):
        """ Return the expected number of weights for the cell given alpha. 
        If alpha is None, it is set to the cell's measured flux minus 1.
        The result scales with S, and depends on weight distribution"""
        if alpha is None:
            alpha = self.flux-1
        return self.S / np.mean( self.w / (1 + alpha * self.w) )

    def randomize(self, cell=None, *, alpha=None, S=None, name='random', random_state=None):
        """ Return a randomized version of the input cell or the template cell with a new set of weights.

        Parameters:
        cell : Cell, optional
            The cell to randomize. If None, the template cell itself is used.
        alpha : float, optional
            The alpha parameter used to adjust the expected number of weights. If None, it is set to the expected flux minus 1.
        S : float, optional
            The scaling factor for the number of weights. If None, it defaults to the cell's S attribute.
        name : str, optional
            The name of the new randomized cell.
        random_state : int or np.random.RandomState, optional
            The random state for reproducibility.

        Notes:
        Weights are drawn from the cumulative weight distribution using uniform probability distribution between 0 and 1

        """
        # Number of weights to generate: drawn from a Poisson distribution with mean mu.
        if cell is None:
            cell = self
        else:
            if S is not None:
                raise ValueError("Cannot specify S when providing a different cell.")
            

        S = cell.S if S is None else S
        mu = cell.expected_n(alpha) * S/self.S
        size = stats.poisson.rvs(mu, random_state=random_state)
        newcell = dict(
            t=cell.t,
            tw=cell.tw,
            n=size,  
            S=S, 
            B=cell.B*S/self.S,
            w=cell.weight_sampler()(stats.uniform.rvs(scale=1, size=size, random_state=random_state)
                                      )
        )
         
        return Cell(newcell,name=name+f'_{random_state}')


    def generate_sim_cells(self, N=10000, S=None, alpha=None, ):
        from concurrent.futures import ThreadPoolExecutor
        import os

        def make_sim_cell(i):
            return self.randomize(S=S,alpha=alpha, random_state=i, name=f"sim_{i}")
        
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() * 2)) as ex:
            sim_cells = list(ex.map(make_sim_cell, range(N)))
        return sim_cells  

    def non_random(self,  name='non_random'):
        """ Return a non-random version of the input cell with a new set of weights, with the same number of weights as the original cell.
        Weights are drawn using uniform distribution between 0 and 1.
        """
        newcell = dict(t=self.t, tw=self.tw, n=self.n, S=self.S, B=self.B)
                
                
        # Number of weights to generate. Use expected flux, adjusted by alpha, from the setup cell. 
        size = self.n 
        w = self.weight_sampler(np.linspace(0, 1, int(size)))
        newcell['w'] = w
        return Cell(newcell, name=name)

    def plot_cumulative(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6,4))
        # w_edges, Fw = self.get_cumulative_w()
        
        # ax.plot(w_edges, Fw, '-', **kwargs)
        # ax.set(xscale='log', xlabel='Weight $w$', ylabel='Cumulative Distribution',
        #     yticks=[0, 0.25, 0.5, 0.75, 1])
        self.weight_sampler.plot(ax=ax, label=kwargs.get('label', None))
        if 'label' in kwargs and kwargs['label']:
            ax.legend()
        return ax


