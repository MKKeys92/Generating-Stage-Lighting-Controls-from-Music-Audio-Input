import pandas as pd
import os
import matplotlib.pyplot as plt

p_dir = 'log/csv_exports'

dirs = os.listdir(p_dir)

for d in dirs:
    p_subdir = os.path.join(p_dir,d)
    if not os.path.isdir(p_subdir):
        continue
    fs =  sorted(os.listdir(p_subdir))
    idx = []
    c = 0
    for f in fs:
        if f.endswith('.csv'):
            if 'loss' in f:
                fig, ax = plt.subplots(figsize=(12, 4))

                p_f = os.path.join(p_dir, d, f)
                df = pd.read_csv(p_f)

                x = df['Step'].values.tolist()
                x = [e / 8 for e in x]
                y = df['Value'].values.tolist()

                ax.plot(x, y)
                ax.set_xlabel("epochs")
                ax.set_ylabel('loss')

                p = os.path.join(p_subdir, d + '_loss_plot.png')
                plt.savefig(p , bbox_inches='tight')
                plt.close(fig)

            else:
                idx.append(c)
        c+=1

    n = len(idx)
    if n == 0:
        continue
    fig, axs = plt.subplots(nrows=n, figsize = (12, 3*(n)), sharex=True)
    k=0
    for i in idx:
        f = fs[i]
        ax = axs[k]
        p_f = os.path.join(p_dir, d, f)
        df = pd.read_csv(p_f)

        x = df['Step'].values.tolist()
        y = df['Value'].values.tolist()

        ax.plot(x,y)
        if i == idx[-1]:
            ax.set_xlabel("epochs")

        if 'RMS' in p_f:
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel('$\Gamma_\mathrm{loud \leftrightarrow bright}$')
        elif 'Max_Beat' in p_f:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('$\Gamma_\mathrm{beat \leftrightarrow peak}$')
        elif 'Min_Beat' in p_f:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('$\Gamma_\mathrm{beat \leftrightarrow valley}$')
        elif 'SSM' in p_f:
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel('$\Gamma_\mathrm{structure}$')
        elif 'Nov_Corr' in p_f:
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel('$\Gamma_\mathrm{novelty}$')
        elif 'MSE' in p_f:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('$\Omega_\mathrm{meansquared}$')
        elif 'PFID' in p_f:
            ax.set_ylim(-0.05, 20.05)
            ax.set_ylabel('$\Omega_\mathrm{frechet}$')
        elif 'Int' in p_f:
            ax.set_ylim(-0.005, 0.1)
            ax.set_ylabel('$\Psi_\mathrm{intensity}$')
        k+=1
    p = os.path.join(p_subdir, d + '_eval_plots.png')
    plt.savefig(p, bbox_inches='tight')
    plt.close(fig)