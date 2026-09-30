"""Vector illustration of the frozen request design; no result data or inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]

def main():
    fig,ax=plt.subplots(figsize=(7.0,3.35))
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
    ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
    colors={'ink':'#162A38','muted':'#4E6070','focal':'#DCECF4','background':'#EDF0F2',
            'base':'#F1D3AA','same':'#BADCCF','view':'#CCD5EB'}
    def text(x,y,s,size=8,weight='normal',ha='center',color='ink'):
        ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va='center',color=colors.get(color,color))
    def box(x,y,w,h,s,fill,fontsize=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.06',lw=.7,ec='#B9C3CA',fc=colors.get(fill,fill)))
        text(x+w/2,y+h/2,s,fontsize)
    def arrow(a,b,color='#7B8993'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.8,color=color))
    text(1.7,4.68,'Fixed records',10,'bold')
    text(5.0,4.68,'Three request pairs',10,'bold')
    text(8.4,4.68,'Fixed decision checks',10,'bold')
    text(1.65,4.25,'Same records and values',8.2,color='muted')
    text(5.0,4.25,'Random F/R launch slots',8.2,color='muted')
    text(8.4,4.25,'Common initial label: F1',8.2,color='muted')
    text(.35,3.45,'F',8,'bold')
    box(.6,3.16,1.25,.59,'Focal','focal');box(1.9,3.16,1.25,.59,'Background','background',7)
    text(.35,2.58,'R',8,'bold')
    box(.6,2.29,1.25,.59,'Background','background',7);box(1.9,2.29,1.25,.59,'Focal','focal')
    text(1.65,1.63,'Hydraulic: 9 + 9 per panel\nBearing: 36 + 12',7.8)
    text(1.65,.62,'Two rounds per panel\nNo reference labels or source IDs',8.2,color='muted')
    arrow((3.3,3.05),(3.78,3.05))
    text(4.35,3.89,'F',8,'bold');text(5.48,3.89,'R',8,'bold')
    for j,y in [(1,3.05),(2,2.18),(3,1.31)]:
        box(3.92,y,.87,.58,f'F{j}','base' if j==1 else ('same' if j==2 else 'background'))
        box(5.04,y,.87,.58,f'R{j}','view' if j==2 else 'background')
        text(6.43,y+.29,f'Pair {j}',7.2,color='muted')
    text(4.97,.62,'F2/R2 checked near the same time\nPair starts 180 s apart',8.2,color='muted')
    arrow((6.7,3.03),(7.13,3.03))
    box(7.23,3.15,2.32,.61,'Same: accept iff F1 = F2','same',7.8)
    box(7.23,2.27,2.32,.61,'View: accept iff F1 = R2','view',7.8)
    text(8.4,1.48,'Otherwise abstain\nMissing required call: abstain',7.7)
    text(8.4,.58,'Errors caught and correct labels lost\nUsable correct output and resources',8.2,color='muted')
    target=ROOT/'paper/figures/policy_design_v1.pdf';fig.savefig(target,bbox_inches='tight',metadata={'Title':'Frozen paired-position and decision-check design'})
    fig.savefig(ROOT/'tmp/pdfs/policy_design_v1.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
