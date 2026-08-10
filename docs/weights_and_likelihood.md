# Weights and Likelihood

## The photon dataset: bands and pixels

## Selecting a source: data subset with weights

## Cells and the Kerr likelihood formula  


As derived in Kerr, For each cell with a set of photons with weights $w$, the log likelihood as a function of $\alpha$  and $\beta$ is

$$ \displaystyle\log\mathcal{L}(\alpha,\beta\ |\ w)\ = \sum_{w}  \log \big( 1 + \alpha\ w + \beta\ (1-w) \big) - (\alpha\ S + \beta\ B) $$

where  $\alpha$ and $\beta$ are the excess signal and background fractions.
$S$ and $B$ are the expected numbers of signal and background counts for the 
cell, determined from the full data set and relative exposure for the cell. 

We will fix $\beta=0$, assuming that no component of the background is varying, and consider only signal variation. Then the expression simplifies to 
$$ \displaystyle\log\mathcal{L}(\alpha |\ w)\ = \sum_{w}  \log \big( 1 + \alpha\ w ) - \alpha\ S  $$

The solution for $\alpha$ must satisfy
$$  \sum_w w/(1+\alpha w) = S $$

In the special case where all $w$ values are 1, this reduces to the Poisson propability distribution, with the
maximum likelihood solution $\alpha = N/S-1$, where $N$ is the number of photons.

