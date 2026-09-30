"""Plot matched seven-case context probes; no model calls and no pooled trials."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/mechanism_extension'

def main():
    data=json.loads((BASE/'analysis/scores.json').read_text())
    pairs=data['paired_contrasts']
    rows=[]
    for model in ['gpt-5.6-sol','gpt-6-astra']:
        for repeat in (1,2,3):
            subset=next(x for x in pairs if x['model']==model and x['repeat']==repeat and x['left']=='clean' and x['right']=='subset_batch')
            single=next(x for x in pairs if x['model']==model and x['repeat']==repeat and x['left']=='clean' and x['right']=='single_panel')
            if subset['paired_n']!=7 or single['paired_n']!=7:
                raise ValueError('This matched plot requires all seven probe cases; missingness needs explicit redesign')
            for context,acc in [('70-record batch',subset['left_accuracy']),('7-record batch',subset['right_accuracy']),('Single-record calls',single['right_accuracy'])]:
                rows.append({'model':model,'repeat':repeat,'context':context,'n_cases':7,'correct':round(acc*7),'accuracy':acc})
    folder=ROOT/'paper/figures/mechanism_extension';folder.mkdir(exist_ok=True,parents=True)
    out=BASE/'analysis/context_figure_data.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.25),sharey=True,constrained_layout=True)
    contexts=['70-record batch','7-record batch','Single-record calls']
    for ax,model,title in zip(axes,['gpt-5.6-sol','gpt-6-astra'],['GPT-5.6 Sol','GPT-6 Astra']):
        for repeat,offset,color,marker in zip((1,2,3),(-.08,0,.08),('#185674','#c77326','#8271a6'),('o','s','^')):
            yy=[next(x['correct'] for x in rows if x['model']==model and x['repeat']==repeat and x['context']==c) for c in contexts]
            ax.plot(np.arange(3)+offset,yy,marker=marker,color=color,linewidth=1.0,alpha=.9,label=f'R{repeat}')
        ax.set_xticks(range(3),['70-record\nbatch','7-record\nbatch','Single-record\ncalls'])
        ax.set_yticks(range(8));ax.set_ylim(-.3,7.5);ax.set_xlim(-.25,2.25)
        ax.set_title(title);ax.grid(axis='y',alpha=.16);ax.legend(loc='lower left',ncol=3,fontsize=8,frameon=False)
    axes[0].set_ylabel('Correct among the same seven cases')
    fig.savefig(folder/'matched_context.pdf');fig.savefig(folder/'matched_context.png',dpi=220);plt.close(fig)
    print('Built matched context figure and 18-row figure data.')

if __name__=='__main__':main()
