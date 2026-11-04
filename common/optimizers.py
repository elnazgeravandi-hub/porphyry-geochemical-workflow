"""Explicit bounded MOPSO and NSGA-II; all requests count, including cached fits."""
import numpy as np
from sklearn.cluster import kmeans_plusplus


def dominates(a,b):
    if a['v'] != b['v']: return a['v'] < b['v']
    if a['v'] > 0: return False
    return bool(np.all(a['f']<=b['f']) and np.any(a['f']<b['f']))


def fronts(pop):
    remaining=list(range(len(pop))); result=[]
    while remaining:
        front=[i for i in remaining if not any(dominates(pop[j],pop[i]) for j in remaining if j!=i)]
        result.append(front); removed=set(front); remaining=[i for i in remaining if i not in removed]
    return result


def crowding(pop, indices):
    out={i:0. for i in indices}
    if len(indices)<=2: return {i:np.inf for i in indices}
    for q in range(2):
        order=sorted(indices,key=lambda i:(pop[i]['f'][q],i))
        lo,hi=pop[order[0]]['f'][q],pop[order[-1]]['f'][q]
        if not np.isfinite([lo,hi]).all() or hi==lo: continue
        out[order[0]]=out[order[-1]]=np.inf
        for pos in range(1,len(order)-1):
            out[order[pos]]+=(pop[order[pos+1]]['f'][q]-pop[order[pos-1]]['f'][q])/(hi-lo)
    return out


def initial_population(evaluator, seed, size):
    rng=np.random.default_rng(seed)
    if evaluator.method=='kmeans':
        return np.array([kmeans_plusplus(evaluator.z,n_clusters=evaluator.k,
            random_state=int(rng.integers(0,2**31-1)))[0].ravel() for _ in range(size)])
    return rng.uniform(evaluator.lo,evaluator.hi,size=(size,len(evaluator.lo)))


def archive_update(archive,candidates,capacity):
    merged=[]; seen=set()
    for rec in archive+candidates:
        key=tuple(rec['f'])
        if rec['v']==0 and key not in seen:
            merged.append(rec); seen.add(key)
    if not merged: return []
    archive=[merged[i] for i in fronts(merged)[0]]
    while len(archive)>capacity:
        distances=crowding(archive,list(range(len(archive))))
        drop=min(distances,key=lambda i:(distances[i],i)); archive.pop(drop)
    return archive


def mopso(ev,initial,seed,generations=25):
    rng=np.random.default_rng(seed); x=initial.copy(); velocity=np.zeros_like(x)
    pop=[ev.evaluate(row) for row in x]; best=pop.copy(); archive=archive_update([],pop,50)
    vmax=.15*(ev.hi-ev.lo)
    for _ in range(generations):
        new=[]
        distances=crowding(archive,list(range(len(archive)))) if archive else {}
        # Synchronous population update; leader archive fixed within each generation.
        for i in range(len(x)):
            if archive:
                a,b=rng.integers(len(archive),size=2)
                winner=a if distances[a]>distances[b] else b if distances[b]>distances[a] else int(rng.choice([a,b]))
                leader=archive[winner]['theta']
            else:
                minimum=min(p['v'] for p in pop)
                choices=[p for p in pop if p['v']==minimum]
                leader=choices[int(rng.integers(len(choices)))]['theta']
            v=.72*velocity[i]+1.49*rng.random(len(ev.lo))*(best[i]['theta']-x[i])+1.49*rng.random(len(ev.lo))*(leader-x[i])
            v=np.clip(v,-vmax,vmax); proposed=x[i]+v
            outside=(proposed<ev.lo)|(proposed>ev.hi)
            x[i]=np.clip(proposed,ev.lo,ev.hi); v[outside]*=-.5; velocity[i]=v
            rec=ev.evaluate(x[i]); new.append(rec)
            if dominates(rec,best[i]) or (not dominates(best[i],rec) and rng.random()<.5): best[i]=rec
        pop=new; archive=archive_update(archive,pop,50)
    return archive


def sbx(a,b,lo,hi,rng,probability=.9,eta=15):
    children=[a.copy(),b.copy()]
    if rng.random()>probability: return children
    for j in range(len(a)):
        if rng.random()>.5 or abs(a[j]-b[j])<=1e-14 or hi[j]==lo[j]: continue
        y1,y2=sorted([a[j],b[j]]); u=rng.random()
        beta=1+2*(y1-lo[j])/(y2-y1); alpha=2-beta**(-(eta+1))
        betaq=(u*alpha)**(1/(eta+1)) if u<=1/alpha else (1/(2-u*alpha))**(1/(eta+1))
        c1=.5*((y1+y2)-betaq*(y2-y1))
        beta=1+2*(hi[j]-y2)/(y2-y1); alpha=2-beta**(-(eta+1))
        betaq=(u*alpha)**(1/(eta+1)) if u<=1/alpha else (1/(2-u*alpha))**(1/(eta+1))
        c2=.5*((y1+y2)+betaq*(y2-y1))
        if rng.random()<.5: c1,c2=c2,c1
        children[0][j]=np.clip(c1,lo[j],hi[j]); children[1][j]=np.clip(c2,lo[j],hi[j])
    return children


def mutate(x,lo,hi,rng,eta=20):
    x=x.copy()
    for j in range(len(x)):
        if rng.random()>1/len(x) or hi[j]==lo[j]: continue
        d1=(x[j]-lo[j])/(hi[j]-lo[j]); d2=(hi[j]-x[j])/(hi[j]-lo[j]); u=rng.random()
        if u<=.5:
            value=2*u+(1-2*u)*(1-d1)**(eta+1); delta=value**(1/(eta+1))-1
        else:
            value=2*(1-u)+2*(u-.5)*(1-d2)**(eta+1); delta=1-value**(1/(eta+1))
        x[j]=np.clip(x[j]+delta*(hi[j]-lo[j]),lo[j],hi[j])
    return x


def nsga2(ev,initial,seed,generations=25):
    rng=np.random.default_rng(seed); size=len(initial); pop=[ev.evaluate(row) for row in initial]
    for _ in range(generations):
        rank={}; distance={}
        for r,front in enumerate(fronts(pop)):
            rank.update({i:r for i in front}); distance.update(crowding(pop,front))
        def tournament():
            a,b=rng.integers(size,size=2)
            ka,kb=(rank[a],-distance[a]),(rank[b],-distance[b])
            i=a if ka<kb else b if kb<ka else int(rng.choice([a,b]))
            return pop[i]['theta']
        children=[]
        while len(children)<size:
            for child in sbx(tournament(),tournament(),ev.lo,ev.hi,rng):
                if len(children)<size: children.append(ev.evaluate(mutate(child,ev.lo,ev.hi,rng)))
        combined=pop+children; selected=[]
        for front in fronts(combined):
            if len(selected)+len(front)<=size: selected.extend(front)
            else:
                distances=crowding(combined,front)
                selected.extend(sorted(front,key=lambda i:(-distances[i],i))[:size-len(selected)]); break
        pop=[combined[i] for i in selected]
    return [pop[i] for i in fronts(pop)[0] if pop[i]['v']==0]


def objective_front(population):
    """First nondominated front for finite, feasible two-objective records.

    O(n log n), retaining identical objective vectors as separate candidates.
    Evolutionary survival still uses the general feasibility-first front sorter.
    """
    indices=[i for i,p in enumerate(population) if p['v']==0 and np.isfinite(p['f']).all()]
    order=sorted(indices,key=lambda i:(population[i]['f'][0],population[i]['f'][1],i))
    result=[];best_second=np.inf;position=0
    while position<len(order):
        stop=position+1;first=population[order[position]]['f'][0]
        while stop<len(order) and population[order[stop]]['f'][0]==first:stop+=1
        group=order[position:stop];minimum=population[group[0]]['f'][1]
        if minimum<best_second:
            result.extend(i for i in group if population[i]['f'][1]==minimum)
        best_second=min(best_second,minimum);position=stop
    return result
