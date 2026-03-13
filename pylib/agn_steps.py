import sys
from pathlib import Path
from pylib.data_setup import VarDB, set_theme
from utilities.ipynb_docgen import show, show_date, show_fig
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def select_single_step(vdb, margin=50, bin_width=7):
    show(f"""### Detect the single-steppers
Here I select BB light curves with exactly two blocks, and record the ratio
of the two levels (the steps), and the position of the transition.
The light curves were generated with a bin width of {bin_width} days, so the "step" position
must be at a bin boundary, the boundary position depending on the BB algorithm, its uncertainty TBD.
Ratios close to 1.0 are possibly an artifact of the BB procedure, this needs study.
""")
    dfx = vdb.dfx
    ass = dfx.association.values
    tss = dfx.ts.values
    names = dfx.index
    def make_df(x):
        if x is None: return None
        return pd.DataFrame.from_dict(x)
    lcs = [make_df(vdb[uw_name]['light_curve']) for uw_name in dfx.uw_name]
    
    dd = dict()
    for name, lc, stype, ts in zip(names, lcs, ass, tss):
        if lc is None or len(lc)!=2: continue
        v = lc.tw.values/7
        sh = lc.flux.values.shape
        if sh == (2,):
            a,b = lc.flux.values #  new
        elif sh ==(2,2):
            a,b = lc.flux.values[:,0] # old, with two "flux" columns
        else:
            raise Exception(f'Bad lc.flux shape: {sh}')
        if (a*b>0) & (v[0]>=margin) & (v[-1]>=margin):
            dd[name] = dict(flux_ratio=b/a, time=v[0], t2=v[-1],
                            ts=ts, association=stype)
    df = pd.DataFrame.from_dict(dd, orient='index') 
    df.loc[:,'log_eflux'] = np.log10(dfx.loc[df.index, 'eflux100'])
    df.loc[:,'bbvar'] = dfx.loc[df.index, 'bbvar']
    df.loc[:,'variability'] = dfx.loc[dfx.index, 'variability']

    show(f"""Apply margin={margin} weeks: <br>Found {len(df)} candidates, with the association categories""") 
    assert len(df)>0, 'Failed to find any?'
    v,n = np.unique(df.association,  return_counts=True)
    show(pd.Series(dict(list(zip(v,n))), name=''))
    return df

def ratio_display(df, margin, ax=None, **kwargs):

    eflux = 10**df.log_eflux
    # sns.set_theme(font_scale=1.2)
    fig, ax = plt.subplots(figsize = (10,8)) if ax is None else (ax.figure, ax)
    r = df.flux_ratio

    sns.scatterplot(df, ax=ax, y=np.log10(r).clip(-1,1), 
                    x='time', hue='association', 
                    **kwargs, edgecolor='none', s=25
                    # size=np.log10(eflux), sizes=(10, 150),
                )
    ax.set(xlim=(-100,1000), xlabel='Step time (Fermi week)', 
        ylabel='Flux Ratio (log scale)')
    ax.axhline(0, color="orange", ls='--',  lw=1, alpha=0.5 )
    ticks = np.log10(np.array([1/8,1/4, 1/2, 1, 2, 4,8]))
    ax.set( yticks=ticks, yticklabels='1/8 1/4 1/2 1 2 4 8'.split())
    ax.axvspan(0, margin, color='yellow', alpha=0.10)
    end = np.mean(df.time+df.t2)
    ax.axvspan(end-margin, end, color='yellow', alpha=0.10)
    ax.legend(loc='upper left', fontsize=12,)
    return fig

def ratio_vs_time(df, dfx, margin, week_lim=None, fignum=1):

    show(f"""## Steps: ratio vs. time for blazars""")

    fig, ax = plt.subplots(figsize = (12,10))
    keep = df.association.apply(lambda a: a in 'bll fsrq bcu'.split())
    r = df[keep].flux_ratio
    
    ratio_display(df[keep], margin, ax=ax,
                  hue_order='bll bcu fsrq'.split(), 
                palette='cyan yellow red'.split())

    ax.plot(420, np.log10(2.60) ,'*',color='crimson', ms=20, label='PKS J2333-2343')
    
    show(fig, fignum=fignum, caption=
             f"""The after/before ratio of the apparent step in the flux, vs its time in Fermi weeks. 
             Colors correspond to the 4FGL-DR4 association assignment. The shaded areas are excluded by the 
             {margin}-week exclusion.
             """)

    prop = lambda t: f'{100* sum(df.association==t)/ sum(dfx.association==t):.0f} %'

    show(f"""Notes:
    * The star is the location, after correction for a nearby blazar, of PKS J2333-2343. Because of that, it was not included.
    * There is a higher proportion of BL Lacs ({prop('bll')}) than FSRQs ({prop('fsrq')})).
    * There are more steps-up ({sum(r>1)}) than steps-down ({sum(r<1)}), and the up-steps concentrate, with higher flux ratios, at the end,
    while the fewer down-steps are similarly concentrated at the start. These can be explained by the hypothesis that we are seeing transitions 
    between discrete states, and the time spent in a higher state is less then the lower one. 
    """)

def pulsar_only(df, margin, fignum=2):
    show(f"""## Study the pulsar set to understand BB-induced background
    We will compare this dataset with those for the other association categories.

    """)
    plt.style.use('dark_background')
    psr = df.query('association=="psr"')
    show( ratio_display(psr, margin), fignum=fignum) 

def nbb_count_ratios(dfx):
    """Scatter plot showing the relative frequencies of 1- and 2-block light curves for
    each asociation type.
    
    """
    def get_unique(sclass, name=''):
        v,n = np.unique(sclass,  return_counts=True)
        df= pd.DataFrame.from_dict(
                dict(list(zip(v,n))), orient='index',
            )
        return df.rename(columns={0:name})

    totals = pd.Series(dfx.groupby('association').size(), name='total')

    ones, twos = dfx.query('nbb==1').copy(), dfx.query('nbb==2').copy()
    nbb1 = get_unique(ones.association, 'nbb=1')
    nbb2 = get_unique(twos.association, 'nbb=2')

    nbbstats = pd.concat([totals, nbb1,nbb2], axis=1, )
    # nbbstats.loc[:,'ratio'] = (nbbstats.iloc[:,1]/nbbstats.iloc[:,0]).round(2)
    # show(nbbstats)
    ratio1 = nbbstats.loc[:,'nbb=1'] / nbbstats.loc[:,'total']
    ratio2 = nbbstats.loc[:,'nbb=2'] / nbbstats.loc[:,'total']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x= ratio1, y=ratio2, s=500, marker='+', color='red')
    ax.set(xlabel='nbb=1 fraction', ylabel='nbb=2 fraction', ylim=(0,0.2), xlim=(0,0.8),
           yticks=np.arange(0,0.21,0.05)
           )
    for x,y, name, lr in zip(ratio1, ratio2,ratio1.index,
                            'lllrll' ):
        ax.text(x, y, name, fontsize=20, color='yellow', ha='right' if lr=='r' else 'left', va='bottom')

    return fig

def block_2_1(dfv, dfss, fignum=3):
    """
    dfv -- the dataframe with the BB information, including the nbb column, and the association column
    dfss -- single-step dataframe including the flux_ratio column, and the association
    """
    show(f"""## Compare numbers of 2-block light curves with the single blocks
    There are two reasons for 2-BB light curves to be  related to the 1-BB ones.
    1) Background: It is a false positive generated by the
    BB algorithm. (We need to examine this possibility with a MC.)
    2) Science: The source must have been quiet for at least 8 years, compared with 15 for 
    the 1-block set. The rates must be part of same variability spectrum.
    
    Here is a plot of the 1- abd 2-block fractions for each of our six 
    association classes:
    """)
    # def get_unique(sclass, name=''):
    #     v,n = np.unique(sclass,  return_counts=True)
    #     df= pd.DataFrame.from_dict(
    #             dict(list(zip(v,n))), orient='index',
    #         )
    #     return df.rename(columns={0:name})

    # # copy this column from df, which has the nbb values, to dfv, which has the association values, so we can group by association  
    # # dfv['nbb'] = df.loc[dfv.index, 'nbb']

    # ones, twos = dfv.query('nbb==1').copy(), dfv.query('nbb==2').copy()
    # nbb1 = get_unique(ones.association, 'nbb=1')
    # nbb2 = get_unique(twos.association, 'nbb=2')
    def get_unique(sclass, name=''):
        v,n = np.unique(sclass,  return_counts=True)
        df= pd.DataFrame.from_dict(
                dict(list(zip(v,n))), orient='index',
            )
        return df.rename(columns={0:name})

    nbb1 = get_unique(dfv.query('nbb==1').association, 'nbb=1')

    show_fig(nbb_count_ratios,dfv)
    
    show(""" ### The distributions of the sizes of the steps.
    The pulsar one should only reflect reason 1.""")
    
    import warnings
    warnings.simplefilter("ignore", FutureWarning)
    
    dfss['log_ratio'] = np.log10(dfss.flux_ratio)
    weights= 1/nbb1.loc[dfss.association].values.T[0]
    dfss['bb1_weight'] = weights
    col_order = 'bll fsrq bcu psr unid other'.split()

    sns.displot(dfss, x='log_ratio', # bins=np.linspace(-1,1,41),
                weights= 'bb1_weight',
                col='association',  col_wrap=3, 
                col_order=col_order,
                height=2.5, aspect=1.5, color='yellow',
                kind= 'kde', #element='step', fill=False,
    
               )
    for ax, col_name in zip(plt.gcf().axes, col_order):
        ax.set( xlabel='Flux ratio (log scale)', ylabel='Scaled counts') #, yscale='log')
        ax.axvline(0, color= "0.7"  , )
        ticks = np.log10(np.array([1/8,1/4, 1/2, 1, 2, 4,8]))
        ax.grid(color='0.5')
        ax.text(0.1, 0.8, col_name, fontsize=14, transform=ax.transAxes)
        ax.set(title='')
    ax.set(xlim=(-1.1,1.1),xticks=ticks, xticklabels='1/8 1/4 1/2 1 2 4 8'.split());
    
    show(ax.figure, fignum=fignum,
         caption="""Flux ratio KDE plots for each association 
    category. Counts are scaled by the inverse of the number of sources with nbb=1,
    so the total represents the fraction of nbb=2 to nbb=1. The pulsars are then 
    an estimate of the contribution of false positives to be applied to the others.
    """)
    show(f"""Notes:
    * The range of the pulsar step sizes is  mostly limited to be less than 2.
    * If the apparent pulsar steps are all spurious, its dristribution
    above shouid be a component of all the others. This does not
    appear to be the case, especially for the unid's.
    * The asymmetry between up- and down-steps is dramatically
    different. It appears from this that the most of the bcu's are 
    BL Lacs, and a large proportion of the unid's are as well.
    
    """)

def hist_bbvar(df):
    """
    Histogram of the BB variabiliy index values
    """
    if 'bbvar' not in df.columns:
        raise ValueError("DataFrame does not contain 'bbvar' column.")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.hist(df.bbvar, bins=np.logspace(0, 2, 41), histtype='step', color='C0', lw=2)
    ax.set(xscale='log', xlabel='bbvar', ylabel='count',
           xticks=np.logspace(0, 2, 3), xticklabels=['1', '10', '100', ])
    return fig

def strongest(df):
    show(f"""---
    ## The 10 strongest up-steppers 
    Look for those with a step in the middle...""")
    bdf = df[df.association.apply(lambda a: a in 'bll fsrq bcu'.split())].sort_values('ts', ascending=False).copy()
    bdfq = bdf.query('flux_ratio>1.5 & 200<time<600')
    show(bdfq.head(10))

    show(f"""---
    ## The 10 strongest down-steppers 
    Look for those with a step in the middle...""")
    bdfd = df[df.association.apply(lambda a: a in 'bll fsrq bcu'.split())].sort_values('ts', ascending=False).copy()
    bdfqd = bdf.query('flux_ratio<1/1.5 & 200<time<600')
    show(bdfqd.head(10))

#==============================================================================

def main(data_version='v3', margin=100):

    show("""# Observation of steps in gamma-ray light curves
        
        By a "step" we mean a change, typically around a factor or two in flux, within a short time, typically a few weeks, that is sustained for a long time, typically years.
        """)
    VarDB.info_file= f'files/source_info_{data_version}.pkl'
    set_theme(['dark'])
    show_date()


    show(f"""## Load light curve data
        Note that the file `{Path(VarDB.info_file).name}`, used here was copied from SLAC's s3df,
        where is was last updated on Jun  6 2025. 
        It can be found at `/sdf/home/b/burnett/work/bb_light_curves/files`.
        """)

    vdb = VarDB().load_cats().matchup()   
    dfv = vdb.dfx

    df = select_single_step(vdb, margin)
    
    show(f""" ## BB variabiilty index
         Each block in the light curves was fit by maximizing the likelihoods. The variability index, 
         `bbvar`, was computed from the likelihod functions exacly as for the catalog. 
         Here is a plot for the 2-block subset.
         """)

    show_fig(hist_bbvar, df, )

    show("""For the following we require `bbvar>10' """)
    dfss = df.query('bbvar>10').copy()

    ratio_vs_time(dfss, dfv, margin)
    pulsar_only(dfss, margin, fignum=2)

    try:

  
        block_2_1(dfv, dfss)

        strongest(dfss)

        show(f"""# Summary
            
        * We have identified a set of 2-block light curves, with steps typically around a factor of two.
        * The steps are not symmetrically distributed in time, with the up-steps concentrated at the end of the mission, and the down-steps at the start.
        * The distribution of step sizes for pulsars, which should be all background, is different from that for blazars, especially for the unid's.
        * There are more BL Lacs than FSRQs among the steppers, and a large proportion of the unid's appear to be BL Lacs as well.
        * The strongest steppers include some well known sources, and some not so well known.   
        """)

    except Exception as e: 
        print(f'Failed to run block_2_1:  {e}')  
        return dfv,dfss

if __name__ == '__main__' and 'main' in sys.argv:
    
    data_version = 'v1' if 'v1' in sys.argv else 'v3'
    
    main(data_version)
