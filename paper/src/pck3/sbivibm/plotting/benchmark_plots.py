import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import os
import uuid

import numpy as np 
import torch 
import pandas as pd

from sbi.analysis import pairplot
from sbibm.visualisation import fig_metric

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from sbivibm.utils import SEPERATOR, get_posteriors, get_predictive_samples_by_id, get_samples, get_metrics, get_full_dataset, PATH, get_samples_by_id, get_predictive_samples, get_predictive_samples_by_id
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import warnings
warnings.filterwarnings("ignore")

METRIC_NAME_DICT = {"c2st":"C2ST", "mmd":"MMD", "median_dist":"Predictive median dist", "ksd":"Kernel Stein Discrepancy", "mvn_pq":"Gaussian KL(p||q)", "mvn_qp":"Gaussian KL(q||p)"}
LOSS_DICT = {"elbo": r"$\mathcal{L}_{rKL}$", "forward_kl":r"$\mathcal{L}_{fKL}$",  r"renjey_divergence ($\alpha = $0.5)":r"$\mathcal{L}_{\alpha}$ ($\alpha = 0.5$)", "iwelbo":r"$\mathcal{L}_{IW}$", "na":"MCMC", "na (100 chains)":"MCMC (100 chains)", "iwelbo (pathwise)": r"$\mathcal{L}_{IW}$ (STL)", "elbo (mixture)": r"$\mathcal{L}_{rKL}$ (MoS)", "elbo (mixture) (implicit)": r"$\mathcal{L}_{rKL}$ (MoS, impl. grad)", "elbo (mixture_gauss) (implicit)": r"$\mathcal{L}_{rKL}$ (MoG, impl. grad)", "forward_kl (mixture) (implicit)": r"$\mathcal{L}_{fKL}$ (MoS)", "forward_kl (mixture_gauss) (implicit)": r"$\mathcal{L}_{fKL}$ (MoG)", "forward_kl (mixture)": r"$\mathcal{L}_{fKL}$ (MoS)",r"renjey_divergence ($\alpha = $0.5) (unbiased)": r"$\mathcal{L}_{\alpha}$ ($\alpha = 0.5$,unbiased)",r"renjey_divergence ($\alpha = $0.1)": r"$\mathcal{L}_{\alpha}$ ($\alpha = 0.1$)",r"renjey_divergence ($\alpha = $0.3)": r"$\mathcal{L}_{\alpha}$ ($\alpha = 0.3$)",r"renjey_divergence ($\alpha = $0.1) (pathwise)":r"$\mathcal{L}_{\alpha}$ ($\alpha = 0.1$,STL)", r"renjey_divergence ($\alpha = $0.1) (unbiased)":r"$\mathcal{L}_{\alpha}$ ($\alpha = 0.1$,unbiased)"}
GROUP_CATEGORIES = {"rKL": ["elbo"], "fKL":["forward_kl"], "KL":["elbo", "forward_kl"], r"$\alpha$":["renjey_divergence"], "IW":["iwelbo"]}
LOSS_TO_COLOR = {"elbo": "#377eb8", "forward_kl":"#984ea3", r"renjey_divergence ($\alpha = $0.5)":"C3", "iwelbo":"C2", "na":"grey", "na (100 chains)":"black", "iwelbo (pathwise)":"#4daf4a",  "elbo (mixture)":"darkblue", r"renjey_divergence ($\alpha = $0.5) (unbiased)":"darkred", r"renjey_divergence ($\alpha = $0.1) (unbiased)":"darksalmon", r"renjey_divergence ($\alpha = $0.3)":"red", r"renjey_divergence ($\alpha = $0.1)":"indianred", "elbo (mixture_gauss) (implicit)": r"indigo", "forward_kl (mixture_gauss) (implicit)": r"purple", r"renjey_divergence ($\alpha = $0.1) (pathwise)":"#e41a1c",  "forward_kl (mixture)": "mediumblue", "elbo (mixture) (implicit)": "indigo", "forward_kl (mixture) (implicit)": r"darkslateblue"}
TASKS = {"two_moons": "Two moons", "slcp":"SLCP", "bernoulli_glm": "Bernoulli GLM", "lotka_volterra": "Lotka Volterra", "gaussian_linear": "Gaussian Linear", "pyloric": "Pyloric STG"}

def plot_samples(name:str, task:str, algorithm:str,num_simulations:int, loss:str = "elbo", num_observation:int = 1, **kwargs):
    samples = get_samples(name, task, algorithm, num_simulations, loss=loss, num_observation=num_observation)
    folders = list(samples.keys())
    samples = list(samples.values())
    if len(samples) > 1:
        print("Multiple equivalent samples obtained, using most recent one...")
    assert len(samples) > 0, "No samples found"
    fig = pairplot_sns(samples[-1][0], algorithm=algorithm, **kwargs)

    return fig

def plot_predictive_samples(name:str, task:str, algorithm:str,num_simulations:int, loss:str = "elbo", num_observation:int = 1, **kwargs):
    samples = get_predictive_samples(name, task, algorithm, num_simulations, loss=loss, num_observation=num_observation)
    folders = list(samples.keys())
    samples = list(samples.values())
    if len(samples) > 1:
        print("Multiple equivalent samples obtained, using most recent one...")
    assert len(samples) > 0, "No samples found"
    fig = pairplot_sns(samples[-1][0], algorithm=algorithm, **kwargs)

    return fig

def plot_samples_by_id(name, id, **kwargs):
    samples = get_samples_by_id(name,id)
    fig = pairplot_sns(samples, **kwargs)
    fig.tight_layout()
    return fig

def plot_predictive_by_id(name,id,**kwargs):
    samples = get_predictive_samples_by_id(name,id)
    fig = pairplot_sns(samples, **kwargs)
    fig.tight_layout()
    return fig

def ode_predictive_plots(samples, task, num_observation, predictives_to_plot=2):
    if task=="lotka_volterra":
        import sbibm
        task = sbibm.get_task("lotka_volterra", summary=None)
        simulator = task.get_simulator()
        x_true = simulator(task.get_true_parameters(num_observation))
        x_pred = simulator(samples)

        fig = plt.figure(figsize=(8,5))
        mean = task.unflatten_data(x_pred).mean(0).squeeze().T
        t = torch.linspace(*task.tspan, 201)
        q1 = task.unflatten_data(x_pred).quantile(0.05,axis=0).squeeze().T
        q9 = task.unflatten_data(x_pred).quantile(0.95,axis=0).squeeze().T
        observation = task.unflatten_data(x_true).squeeze().T
        obs = plt.plot(t,observation, "--", c="black", label="Observation")
        plt.plot(t,mean, alpha=0.8)
        plt.fill_between(t,q1[:,0],q9[:,0], alpha=0.5)
        plt.fill_between(t,q1[:,1],q9[:,1], alpha=0.5)
        plt.xlabel("Time [s]")
        plt.ylabel("Biomass")
        

        patch1 = mpatches.Patch(color='C0', label='Prey')
        patch2 = mpatches.Patch(color='C1', label='Predator')
        line1 = mlines.Line2D([0], [0],color="black", label="Observation", linestyle="--")
        plt.legend(handles=[patch1, patch2,line1])

        return fig


    elif task=="pyloric":
        from sbivibm.tasks import Pyloric
        task = Pyloric()
        theta = task.get_true_parameters(num_observation)
        seed = task._get_observation_seed(num_observation)
        simulator = task.get_simulator(seed=seed)
        
        # Observations
        voltages = task.unflatten_data(simulator(theta)).squeeze()

        # Predictives
        simulator = task.get_simulator(seed=seed)
        x_pred = simulator(samples[:predictives_to_plot, :])

        # Plotting
        fig, axes = plt.subplots(3,1, figsize=(20,6), sharex=True)

        for i in range(predictives_to_plot):
            x = x_pred[i]
            axes[0].plot(task.t, x[0].numpy().flatten(), linewidth=0.5, alpha=0.5, c="grey")
            axes[1].plot(task.t, x[1].numpy().flatten(), linewidth=0.5, alpha=0.5, c="grey")
            axes[2].plot(task.t, x[2].numpy().flatten(), linewidth=0.5, alpha=0.5, c="grey")

        axes[0].plot(task.t, voltages[0].numpy().flatten(), c="black", linewidth=0.5)
        axes[1].plot(task.t, voltages[1].numpy().flatten(), c="black", linewidth=0.5)
        axes[2].plot(task.t, voltages[2].numpy().flatten(), c="black", linewidth=0.5)

        axes[0].set_ylabel("AB/PD")
        axes[1].set_ylabel("LP")
        axes[2].set_ylabel("PY")

        return fig
    else:
        raise ValueError("Unknown task")



def pairplot_sns(samples, algorithm:str="",  ref_samples=None, color_palette=["black", "red"], true_value=None, diag_plot="hist", non_diag_plot="scatter", reference_upper_corner=True, height=3, limits=None):
    """Pairplot of a distribution, all 1d and 2d marginals...

    Args:
        algorithm (str): [description]
        samples ([type]): [description]
        ref_samples ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    df = pd.DataFrame(samples.numpy(), columns=["dim "+ str(i) for i in range(samples.shape[-1])])
    df["algorithm"] = [algorithm]*len(df)
    if true_value is not None:
        true_value = pd.DataFrame(true_value,  columns=["dim "+ str(i) for i in range(samples.shape[-1])])

    if ref_samples is not None:
        df_ref = pd.DataFrame(ref_samples.numpy(), columns=["dim "+ str(i) for i in range(samples.shape[-1])])
        df_ref["algorithm"] = ["reference"]*len(df_ref)
        df_full = df.append(df_ref)
    else:
        df_full = df

    if ref_samples is not None:
        num_colors = 2
    else:
        num_colors = 1

    COLORS = sns.color_palette(color_palette)[:num_colors]
    if ref_samples is not None:
        def reference_histplot(x,y,color, **kwargs):
            
            x = df_ref[x.name]
            y = df_ref[y.name]
            if non_diag_plot=="hist":
                sns.histplot(x=x,y=y, color=COLORS[1],**kwargs)
            elif non_diag_plot=="scatter":
                sns.scatterplot(x=x,y=y, marker="+",color=COLORS[1],s=2,alpha=0.5)
            else:
                raise NotImplementedError()

    def algorithm_histplot(x,y,color, **kwargs):
        x = df[x.name]
        y = df[y.name]
        if non_diag_plot=="hist":
            sns.histplot(x=x,y=y, color=COLORS[0],**kwargs)
        elif non_diag_plot=="scatter":
            sns.scatterplot(x=x,y=y, marker="+",color=COLORS[0], s=2,alpha=0.5, rasterized=True)
        else:
            raise NotImplementedError()
        

    def true_value_plot_nondiag(x,y,color, **kwargs):
        x = true_value[x.name]
        y = true_value[y.name]
        plt.scatter(x=x,y=y, color="r", rasterized=True)

    def true_value_plot_diag(x,color, **kwargs):
        x = true_value[x.name]
        plt.axvline(x[0], c="r")

    g = sns.PairGrid(data=df_full, hue="algorithm", palette=COLORS, corner=(ref_samples is None or not reference_upper_corner), height=height)
    _ = g.map_lower(algorithm_histplot, bins=100, stat="density")
    if not ref_samples is None:
        if reference_upper_corner:
            g = g.map_upper(reference_histplot, bins=100, stat="density")
        else:
            g = g.map_lower(reference_histplot, bins=100, stat="density")
    _ = g.map_diag(sns.histplot, bins=50, alpha=0.5, stat="density", element="step", fill=False)

    if true_value is not None:
        _ = g.map_diag(true_value_plot_diag)
        _ = g.map_lower(true_value_plot_nondiag)
        if ref_samples is not None and reference_upper_corner:
            _ = g.map_upper(true_value_plot_nondiag)

    if limits is not None:
        for i in range(len(g.axes)):
            for j in range(len(g.axes[i])):
                if g.axes[i,j] is not None:
                    g.axes[i,j].set_ylim(limits[0], limits[1])
                    g.axes[i,j].set_xlim(limits[0], limits[1])

    if num_colors == 2:
        patch1 = mpatches.Patch(color=COLORS[0], label=algorithm)
        patch2 = mpatches.Patch(color=COLORS[1], label="Reference")
        if true_value is None:
            g.axes[0,0].legend(handles=[patch1, patch2])
        else:
            patch3 = mpatches.Patch(color="red", label="True")
            g.axes[-1,-1].legend(handles=[patch1, patch2,patch3], loc="upper center")
    return g.fig




def plot_metrics(name:str, task:str, algorithm="NL", metrics:list=["c2st","mmd"], color_palette=None, groups=["KL", "IW", r"$\alpha$"], fontsize=11, title_fontsize=15, legend_anchor=(2, -0.6), legend_cols=6, legend=True, filter=None, height=3, aspect=1):
    """Plots a metric for all algorithm's and losses

    Args:
        name (str): Benchmark name
        task (str): Task name
        metrics (list, optional): Which metrics to plot. Defaults to ["c2st","mmd"].
        color_palette (str, optional): Color palette . Defaults to None.
    """

    # Reformate df, restric metrics
    df = get_full_dataset(name)
    df = df.query(f"task == '{task}'")

    df = df[df["algorithm"].str.contains(algorithm)]

    # Get MMD max for plotting
    max_mmd = df.groupby("loss").quantile(0.9)["mmd"].max() 

    dfs = []
    index = []
    for met in metrics:
        dfs.append(df[["algorithm", "num_simulations", "loss",met, "parameters", "num_rounds"]])
        index += [met]*len(df)
    df = pd.concat(dfs)
    df["metric"] = index


    for group in groups:
        for loss in GROUP_CATEGORIES[group]:
            mask =  (df["loss"] == loss)
            df["algorithm"][mask] = [algo + "-" + group for algo in df["algorithm"][mask].tolist()]
            
    mask_parameters = (df["parameters"].str.contains("'reduce_variance': True"))
    mask_parameters2 = (df["parameters"].str.contains("'unbiased': True"))
    mask_parameters3 = (df["parameters"].str.contains("'num_components': [2-9]", regex=True)* df["parameters"].str.contains("'flow': 'spline_autoregressive'", regex=True))
    mask_parameters4 = (df["parameters"].str.contains("'num_components': [2-9]", regex=True) * df["parameters"].str.contains("'flow': 'affine_tril'", regex=True))
    mask_rsample = (df["parameters"].str.contains("'rsample': True", regex=True))
    mask_renjey = (df["loss"].str.contains("renjey_divergence")) 
    mask_chains = (df["parameters"].str.contains("'num_chains': [1-9][0-9]", regex=True))
    
    mask_ir = (df["parameters"].str.contains("'sampling_method': 'ir'")) 
    mask_imh = (df["parameters"].str.contains("'sampling_method': 'imh'")) 
    
    df["loss"][mask_renjey] = [data.loss + " " + rf"($\alpha = ${(eval(data.parameters))['alpha']})" for i, data in df[["loss", "parameters"]][mask_renjey].iterrows()]
    df["loss"][mask_chains] = [data.loss + " " + rf"({(eval(data.parameters))['num_chains']} chains)" for i, data in df[["loss", "parameters"]][mask_chains].iterrows()]
    df["loss"][mask_parameters] = [loss + " " + "(pathwise)" for loss in df["loss"][mask_parameters].tolist()]
    df["loss"][mask_parameters2] = [loss + " " + "(unbiased)" for loss in df["loss"][mask_parameters2].tolist()]
    df["loss"][mask_parameters3] = [loss + " " + "(mixture)" for loss in df["loss"][mask_parameters3].tolist()]
    df["loss"][mask_parameters4] = [loss + " " + "(mixture_gauss)" for loss in df["loss"][mask_parameters4].tolist()]
    df["loss"][mask_rsample] = [loss + " " + "(implicit)" for loss in df["loss"][mask_rsample].tolist()]

    if filter is not None:
            df = filter(df)
    # Defined order
    def key_function(val):
        if "NLMCMC" == val or "NRMCMC" == val:
            return 0
        elif "SNLMCMC" == val or "SNRMCMC" == val:
            return 1
        elif "NLVI" == val or "NRVI" == val:
            return 2
        elif "SNLVI" == val or "SNRVI" == val:
            return 3
        elif "NLVI-KL" == val or "NRVI-KL" == val:
            return 4
        elif "SNLVI-KL" == val or "SNRVI-KL" == val:
            return 5
        elif "SNLVI-rKL" == val or "SNRVI-rKL" == val:
            return 5.1
        elif "SNLVI-fKL" == val or "SNRVI-fKL" == val:
            return 5.2
        elif "NLVI-IW" == val or "NRVI-IW" == val:
            return 6
        elif "SNLVI-IW" == val or "SNRVI-IW" == val:
            return 7
        elif r"NLVI-$\alpha$" == val or "NRVI-$\alpha$" == val:
            return 8
        elif r"SNLVI-$\alpha$" == val or "SNRVI-$\alpha$" == val:
            return 9
        elif "c2st" == val:
            return 100
        elif "mmd" == val:
            return 200
        else:
            return 10
    
    
    df = df.sort_values(["metric", "algorithm"], key=lambda x: list(map(key_function, x)))

    def pointplot(x,y,color, **kwargs):
        idx = x.index
        
        ir_idx = mask_ir.index[mask_ir]
        imh_idx = mask_imh.index[mask_imh]
        
        idx_ir = [i for i in idx if i in ir_idx]
        idx_imh = [i for i in idx if i in imh_idx]
        
        if len(idx_ir) > 0:
            sns.pointplot(x=x[idx_ir],y=y[idx_ir], color=color,**kwargs)
        if len(idx_imh) > 0:
            sns.pointplot(x=x[idx_imh],y=y[idx_imh], color=color, linestyles="--",**kwargs)
            return
        idx_both = idx_ir + idx_imh
        remaining_idx = [i for i in idx if i not in idx_both]
        is_mcmc = "MCMC" in df.loc[idx]["algorithm"].tolist()[0]
        if not is_mcmc:
            sns.pointplot(x=x[remaining_idx],y=y[remaining_idx], color=color, linestyles=":", **kwargs)
        else:
            sns.pointplot(x=x[remaining_idx],y=y[remaining_idx], color=color, **kwargs)
    # Get lists to plot
    with sns.axes_style("whitegrid", rc={"font.size":fontsize,"axes.titlesize":title_fontsize,"axes.labelsize":fontsize}):
        METRIC = df.metric.unique()
        LOSSES = df.loss.unique()
        if color_palette==None:
            color_palette = list(map(lambda x: LOSS_TO_COLOR[x], LOSSES))
        COLORS = sns.color_palette(color_palette)
        NUM_SIMULATIONS = df.num_simulations.unique()
        NUM_SIMULATIONS.sort()
        g = sns.FacetGrid(df, col="algorithm", row="metric", hue="loss", sharey=False, palette=COLORS, legend_out=True, height=height, aspect=aspect)
        for met in METRIC:
            g.map(pointplot, "num_simulations", met, order=NUM_SIMULATIONS, capsize=.1, errwidth=1., markers=".", linewidth=0.1, ci=95, scale=0.5)

        ALGORITHM = [g.axes[0,i].get_title().split()[-1] for i in range(len(g.axes[0]))]
        #Better labels...
        
        g.fig.suptitle(TASKS[task], fontsize=title_fontsize)
        for i, metric in enumerate(METRIC):
            g.axes[i,0].set_ylabel(METRIC_NAME_DICT[metric])
            for j, algo in enumerate(ALGORITHM):
                if j > 0:
                    g.axes[i,j].get_yaxis().set_ticklabels([])
                if metric=="c2st":
                    g.axes[i,j].set_ylim(0.5,1)
                    g.axes[i,0].set_ylabel("C2ST", fontsize=fontsize)
                    g.axes[i,j].set_yticks([0.5,0.6, 0.7, 0.8, 0.9,1])
                    g.axes[i,j].tick_params(axis="y", labelsize=fontsize)
                if metric=="mmd":
                    g.axes[i,j].set_ylim(0,max_mmd)
                    g.axes[i,0].set_ylabel("MMD", fontsize=fontsize)
                    if i == 1:
                        g.axes[1,j].set_title("")
                    g.axes[i,j].locator_params(axis="y", nbins=5)
                    g.axes[i,j].tick_params(axis="y", labelsize=fontsize)
                # Title and axis...
                g.axes[0,j].set_title(algo, fontsize=title_fontsize)
                g.axes[-1,j].set_xlabel("Simulations", fontsize=fontsize)
                g.axes[-1,j].set_xticks([0,1,2])
                g.axes[-1,j].set_xticklabels([rf"$10^{int(np.log10(i))}$" for i in NUM_SIMULATIONS], fontsize=fontsize)
        if legend:
            lines = []
            if mask_ir.sum() > 0:
                line = mlines.Line2D([0,1],[0,1],linestyle=':', color="black", label="SIR")
                lines.append(line)
            if mask_imh.sum() > 0:
                line = mlines.Line2D([0,1],[0,1],linestyle='--', color="black", label="IMH")
                lines.append(line)
            g.axes[-1,0].legend(handles=lines + [mpatches.Patch(color=COLORS[i], label=LOSS_DICT[LOSSES[i]]) for i in range(len(LOSSES))],ncol=legend_cols, loc="lower left", bbox_to_anchor=legend_anchor, fontsize=fontsize)
        g.fig.tight_layout()    
        g.fig.subplots_adjust(top=0.85, wspace=0, hspace=0.2)
        sns.despine(left=True)
    return g


def plot_runtimes(name:str, color_palette=None, fontsize=11, title_fontsize=15, legend_anchor=(-0.1, -0.6), legend_cols=6, legend=True, filter=None, quantile_cutoff=0.95, height=3, aspect=1.):
    df = get_full_dataset(name)
    df = df[["algorithm", "task", "num_simulations", "loss","time", "parameters", "num_rounds"]]

    

    mask_NL = df["algorithm"].str.contains("^(?!S)NL.*", regex=True)
    mask_SNL = df["algorithm"].str.contains("SNL.*", regex=True)
    mask_NR = df["algorithm"].str.contains("^(?!S)NR.*", regex=True)
    mask_SNR = df["algorithm"].str.contains("SNR.*", regex=True)
    df["algorithm"][mask_SNL] = "SNL"
    df["algorithm"][mask_NL] = "NL"
    df["algorithm"][mask_SNR] = "SNR"
    df["algorithm"][mask_NR] = "NR"

    mask_parameters = (df["parameters"].str.contains("'reduce_variance': True"))
    mask_parameters2 = (df["parameters"].str.contains("'unbiased': True"))
    mask_parameters3 = (df["parameters"].str.contains("'num_components': [2-9]", regex=True) * df["parameters"].str.contains("'flow': 'spline_autoregressive'", regex=True))
    mask_parameters4 = (df["parameters"].str.contains("'num_components': [2-9]", regex=True) * df["parameters"].str.contains("'flow': 'affine_tril'", regex=True))
    mask_rsample = (df["parameters"].str.contains("'rsample': True", regex=True))
    mask_renjey = (df["loss"].str.contains("renjey_divergence")) 
    mask_chains = (df["parameters"].str.contains("'num_chains': [1-9][0-9]", regex=True))
    
    mask_ir = (df["parameters"].str.contains("'sampling_method': 'ir'")) 
    mask_imh = (df["parameters"].str.contains("'sampling_method': 'imh'")) 
    
    
    df["loss"][mask_chains] = [data.loss + " " + rf"({(eval(data.parameters))['num_chains']} chains)" for i, data in df[["loss", "parameters"]][mask_chains].iterrows()]
    df["loss"][mask_renjey] = [data.loss + " " + rf"($\alpha = ${(eval(data.parameters))['alpha']})" for i, data in df[["loss", "parameters"]][mask_renjey].iterrows()]
    df["loss"][mask_parameters] = [loss + " " + "(pathwise)" for loss in df["loss"][mask_parameters].tolist()]
    df["loss"][mask_parameters2] = [loss + " " + "(unbiased)" for loss in df["loss"][mask_parameters2].tolist()]
    df["loss"][mask_parameters3] = [loss + " " + "(mixture)" for loss in df["loss"][mask_parameters3].tolist()]
    df["loss"][mask_parameters4] = [loss + " " + "(mixture_gauss)" for loss in df["loss"][mask_parameters4].tolist()]
    df["loss"][mask_rsample] = [loss + " " + "(implicit)" for loss in df["loss"][mask_rsample].tolist()]

    def key_function(val):
        if "NL" == val:
            return 0
        elif "SNL" == val:
            return 1
        elif "NR" == val:
            return 2
        elif "SNR" == val:
            return 3
        elif "two_moons" == val:
            return 100
        elif "slcp" == val:
            return 200
        elif "bernoulli_glm" == val:
            return 300
        elif "lotka_volterra" == val:
            return 400
        else:
            return -1

    df = df.sort_values(["task", "algorithm"], key=lambda x: list(map(key_function, x)))
    

    if filter is not None:
            df = filter(df)
            mask_ir = (df["parameters"].str.contains("'sampling_method': 'ir'")) 
            mask_imh = (df["parameters"].str.contains("'sampling_method': 'imh'")) 

    def pointplot(x,y,color, **kwargs):
        idx = x.index
        
        ir_idx = mask_ir.index[mask_ir]
        imh_idx = mask_imh.index[mask_imh]
        
        idx_ir = [i for i in idx if i in ir_idx]
        idx_imh = [i for i in idx if i in imh_idx]
        
        if len(idx_ir) > 0:
            sns.pointplot(x=x[idx_ir],y=y[idx_ir], color=color, linestyles=":",**kwargs)
        if len(idx_imh) > 0:
            sns.pointplot(x=x[idx_imh],y=y[idx_imh], color=color, linestyles="--",**kwargs)
            return
        idx_both = idx_ir + idx_imh
        remaining_idx = [i for i in idx if i not in idx_both]
        sns.pointplot(x=x[remaining_idx],y=y[remaining_idx], color=color,**kwargs)

    with sns.axes_style("whitegrid", rc={"font.size":fontsize,"axes.titlesize":title_fontsize,"axes.labelsize":fontsize}):
        sns.set(font="DejaVu Sans")
        LOSSES = df.loss.unique()
        if color_palette==None:
            color_palette = list(map(lambda x: LOSS_TO_COLOR[x], LOSSES))
        COLORS = sns.color_palette(color_palette)
        ALGORITHM = df["algorithm"].unique()
        TASK = df["task"].unique()
        NUM_SIMULATIONS = df.num_simulations.unique()
        NUM_SIMULATIONS.sort()

        g = sns.FacetGrid(df, row="algorithm", col="task", hue="loss", sharey=False, sharex=True, legend_out=True, palette=COLORS, height=height, aspect=aspect)
        g.map(sns.pointplot, "num_simulations", "time", capsize=.1, order=NUM_SIMULATIONS, errwidth=1., markers=".", linewidth=0.5)

        for i in range(len(g.axes)):
            g.axes[i,0].set_ylabel(f"{ALGORITHM[i]} runtime [s]")
            for j in range(len(g.axes[0])):
                g.axes[0,j].set_title(TASKS[TASK[j]], fontsize=title_fontsize)
                times_q = df.query(f"algorithm=='{ALGORITHM[i]}'").query(f"task=='{TASK[j]}'")["time"].quantile(quantile_cutoff) + 500
                g.axes[i,j].set_ylim(0, times_q)
                if i > 0:
                    g.axes[i,j].set_title("")
                g.axes[-1,j].set_xlabel("Simulations")
                g.axes[-1,j].set_xticks([0,1,2])
                g.axes[-1,j].set_xticklabels([rf"$10^{int(np.log10(i))}$" for i in NUM_SIMULATIONS])
        if legend:
            lines = []
            if mask_ir.sum() > 0:
                line = mlines.Line2D([0,1],[0,1],linestyle=':', color="black", label="SIR")
                lines.append(line)
            if mask_imh.sum() > 0:
                line = mlines.Line2D([0,1],[0,1],linestyle='--', color="black", label="IMH")
                lines.append(line)
            g.axes[-1,0].legend(handles=lines + [mpatches.Patch(color=COLORS[i], label=LOSS_DICT[LOSSES[i]]) for i in range(len(LOSSES))],ncol=legend_cols, loc="lower left", bbox_to_anchor=legend_anchor, fontsize=fontsize)
        
        g.fig.tight_layout()    
    return g

def plot_samples(name, task, num_observation=1, algorithm="SNL", legend=True, limits=None, fontsize=11, title_fontsize=15, filter=None, height=1.5):
    df = get_full_dataset(name)
    df = df.query(f"task == '{task}'").query(f"num_observation == {num_observation}")
    mask_NL = df["algorithm"].str.contains("^(?!S)NL.*", regex=True)
    mask_SNL = df["algorithm"].str.contains("SNL.*", regex=True)
    mask_NR = df["algorithm"].str.contains("^(?!S)NR.*", regex=True)
    mask_SNR = df["algorithm"].str.contains("SNR.*", regex=True)
    if algorithm == "NL":
        df = df[mask_NL]
    elif algorithm == "SNL":
        df = df[mask_SNL]
    elif algorithm == "NR":
        df = df[mask_NR]
    elif algorithm == "SNR":
        df = df[mask_SNR]
    else:
        df = df[df["algorithm"] == algorithm]
    df = df[["num_simulations", "loss", "parameters", "folder"]]

    mask_parameters = (df["parameters"].str.contains("'reduce_variance': True"))
    mask_parameters2 = (df["parameters"].str.contains("'unbiased': True"))
    mask_parameters3 = (df["parameters"].str.contains("'num_components': [2-9]", regex=True) * df["parameters"].str.contains("'flow': 'spline_autoregressive'", regex=True))
    mask_parameters4 = (df["parameters"].str.contains("'num_components': [2-9]", regex=True) * df["parameters"].str.contains("'flow': 'affine_tril'", regex=True))
    mask_rsample = (df["parameters"].str.contains("'rsample': True", regex=True))
    mask_renjey = (df["loss"].str.contains("renjey_divergence")) 
    mask_chains = (df["parameters"].str.contains("'num_chains': [1-9][0-9]", regex=True))
    
    mask_ir = (df["parameters"].str.contains("'sampling_method': 'ir'")) 
    mask_imh = (df["parameters"].str.contains("'sampling_method': 'imh'")) 
    
    
    df["loss"][mask_chains] = [data.loss + " " + rf"({(eval(data.parameters))['num_chains']} chains)" for i, data in df[["loss", "parameters"]][mask_chains].iterrows()]
    df["loss"][mask_renjey] = [data.loss + " " + rf"($\alpha = ${(eval(data.parameters))['alpha']})" for i, data in df[["loss", "parameters"]][mask_renjey].iterrows()]
    df["loss"][mask_parameters] = [loss + " " + "(pathwise)" for loss in df["loss"][mask_parameters].tolist()]
    df["loss"][mask_parameters2] = [loss + " " + "(unbiased)" for loss in df["loss"][mask_parameters2].tolist()]
    df["loss"][mask_parameters3] = [loss + " " + "(mixture)" for loss in df["loss"][mask_parameters3].tolist()]
    df["loss"][mask_parameters4] = [loss + " " + "(mixture_gauss)" for loss in df["loss"][mask_parameters4].tolist()]
    df["loss"][mask_rsample] = [loss + " " + "(implicit)" for loss in df["loss"][mask_rsample].tolist()]

    if filter is not None:
        df = filter(df)

    def key_function(val):
        if "na" in val:
            return 0
        elif "elbo" in val and "iw" not in val:
            return 1
        elif "forward" in val:
            return 2
        elif "renjey" in val:
            return 3
        elif "iwelbo" in val:
            return 4
        else:
            return 10
    
    df = df.sort_values(["loss"], key=lambda x: list(map(key_function, x)))

    with sns.axes_style("white", rc={"font.size":fontsize,"axes.titlesize":title_fontsize,"axes.labelsize":fontsize}):
        LOSSES = df.loss.unique()
        NUM_SIMULATIONS = df.num_simulations.unique()
        NUM_SIMULATIONS.sort()

        event_dim = 0
        samples = dict()
        for folder in df["folder"].tolist():
            samples[folder] = get_samples_by_id(name, folder)
            event_dim = samples[folder].shape[-1]

        def algorithm_histplot(x,y,color, **kwargs):
            df_new = df[df[x.name] == x.tolist()[0]]
            df_new = df_new[df_new[y.name] == y.tolist()[0]]
            folder = df_new["folder"].tolist()[0]
            sample = samples[folder]
            if event_dim == 1:
                #plt.hexbin(sample[:,0], sample[:,1], extent=[-1,1,-1,1], bins=500, cmap="viridis")
                #sns.scatterplot(x=sample[:,0],y=sample[:,1], marker="+",color="black",s=2,alpha=0.5)
                pass
                
            else:
                fig = pairplot_sns(sample, height=height, limits=limits)
                plt.close()
                #fig.set_dpi(1)
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                ax.set_adjustable("box", share=False)
                canvas.draw()
                image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.imshow(image)

        g = sns.FacetGrid(df, col="loss", row="num_simulations", sharey=False, sharex=False, legend_out=True)
        g.map(algorithm_histplot, "loss","num_simulations")

        for i in range(len(g.axes)):
            g.axes[i,0].set_ylabel(rf"Simulations: $10^{int(np.log10(NUM_SIMULATIONS[i]))}$", fontsize=fontsize)
            for j in range(len(g.axes[0])):
                if i > 0:
                    g.axes[i,j].set_title("")
                else:
                    g.axes[i,j].set_title(LOSS_DICT[LOSSES[j]])
                g.axes[i,j].set_xticks([])
                g.axes[i,j].set_yticks([])
                g.axes[i,j].set_xlabel("")
        g.fig.tight_layout()
        g.fig.subplots_adjust(wspace=0, hspace=0)
        sns.despine(left=True, bottom=True)
    return g

def runtime_plot(name:str, task:str):
    """ Runtime plot

    Args:
        name (str): Benchmark name
        task (str): The benchmark task

    Returns:
        [axes]: Matplotlib axes
    """
    # TODO loss filters
    df = get_full_dataset(name)
    df = df.query(f"task == '{task}'")
    df["loss"]= df["loss"].map(LOSS_DICT)
    NUM_SIMULATIONS = df.num_simulations.unique()
    ax = sns.pointplot(df["num_simulations"],df["time"], hue=df["loss"])
    ax.legend_.set_title(None)
    ax.set_xticks(list(range(len(NUM_SIMULATIONS))))
    ax.set_xticklabels([rf"$10^{int(np.log10(i))}$" for i in NUM_SIMULATIONS])
    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Number of Simulations")
    return ax


def plot_sbibm_plot(name:str, task:str, metric):
    df = get_full_dataset(name)
    df = df.query(f"task == '{task}'")
    df["algorithm"] = df["algorithm"] + df["loss"]
    fig = fig_metric(df, metric=metric)
    return fig


