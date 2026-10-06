"""Run a fully chronological neural-memory benchmark with independent risk probes.

Example: python -m nonstationarity.experiment --config configs/full.json --out results/full --jobs 4
The replication unit is a complete independently generated stream, not a time point.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import argparse
import hashlib
import json
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from .dgp import Generator, SCENARIOS, STRESS
from .kernels import candidate_grid, ExpertSpec
from .neural import NeuralBank
from .selectors import Prequential, FixedShare, comparison_penalties
from .meta import AdaHedge, ShareGridAdaHedge, ExtendedExpertPool, sparse_vector
from .baselines import RLS, MazzettoUpfalLinear, DriftMonitors, RiverModels


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def masks_for(specs):
    def ids(test):
        return np.array([i for i, s in enumerate(specs) if test(s)], dtype=int)
    expand = lambda s: s.kind == "expanding"
    power = lambda s: s.kind in ("power", "capped", "expanding")
    return {
        "Uniform_CV": ids(lambda s: s.kind == "uniform" or expand(s)),
        "Exponential_CV": ids(lambda s: s.kind == "exponential" or expand(s)),
        "Power_CV_matched": ids(lambda s: (s.kind == "power" and s.shape == 1.) or expand(s)),
        "Joint_CV": ids(power),
        "Joint_1SE": ids(power),
        "Width_only_CV": ids(expand),
        "Memory_only_CV": ids(lambda s: power(s) and s.width == 12),
    }


def run_stream(config, scenario, seed, out_dir, supplied=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    threadpool_limits(1)
    started = time.perf_counter()
    T, d = int(config["T"]), int(config.get("d", 4))
    start, burn = int(config.get("start",64)), int(config.get("burnin",128))
    stride = int(config.get("refit_stride",32))
    nprobe = int(config.get("probes",256))
    generator = Generator(scenario, T, d, seed)
    stream = generator.sample() if supplied is None else supplied
    X, y = stream.X, stream.y
    if X.shape != (T,d) or len(y) != T:
        raise ValueError("input shape mismatch")
    specs = candidate_grid(config.get("histories",[32,96,256,768]), config.get("widths",[4,12,24]))
    K = len(specs)
    masks = masks_for(specs)
    use_river = bool(config.get("river", True))
    # Dynamic neural wrappers are never selectable by joint CV/aggregation.
    if use_river:
        specs_all = specs + [ExpertSpec("uniform", 32, 1., 12), ExpertSpec("uniform",32,1.,12)]
        detectors, online = DriftMonitors(), RiverModels(seed)
    else:
        specs_all, detectors, online = specs, None, None
    bank = NeuralBank(specs_all, d, seed, steps=int(config.get("steps",60)),
                      lr=float(config.get("lr",.03)), freeze_hidden=config.get("freeze_hidden",False))
    memories = config.get("selection_memories", {})
    selectors = {name: Prequential(specs, int(memories.get(name,96)), 1. if name == "Joint_1SE" else 0.)
                 for name in masks}
    # Scale calibration consumes the initial observed segment only.
    variance_scale = max(.05, float(np.var(y[:start])))
    score_scale = 4*variance_scale
    mixes = {
        "Joint_FixedShare": FixedShare(K, eta=1., share=.01, scale=score_scale, active=masks["Joint_CV"]),
        "Uniform_FixedShare": FixedShare(K, eta=1., share=.01, scale=score_scale, active=masks["Uniform_CV"]),
        "Hedge_all": FixedShare(K, eta=1., share=0., scale=score_scale),
    }
    rls = {"RLS_099": RLS(d,.99), "RLS_0995": RLS(d,.995)}
    # Second-stage expert pool. Shrinkage creates no extra NN fits.
    ext_pool = ExtendedExpertPool(specs, tuple(config.get("shrinkages",[.25,.5,.75,1.])))
    meta_scale = score_scale
    ada_nn = AdaHedge(K)
    ada_ext = AdaHedge(ext_pool.size)
    ada_prior = AdaHedge(ext_pool.size, prior=ext_pool.complexity_prior())
    share_nn = ShareGridAdaHedge(K,meta_scale)
    share_ext = ShareGridAdaHedge(ext_pool.size,meta_scale)
    share_prior = ShareGridAdaHedge(ext_pool.size,meta_scale,prior=ext_pool.complexity_prior())
    mu = {"MU_linear_conservative": MazzettoUpfalLinear(d),
          "MU_linear_tuned": MazzettoUpfalLinear(d, threshold_scale=config.get("mu_threshold_scale",.01))}
    # All baselines receive every historical row; none sees t before predicting t.
    for s in range(start):
        for model in rls.values(): model.update(X[s],y[s])
        if online: online.update(X[s],y[s])
    index = {(s.kind,s.h,s.width,s.shape,s.cap):i for i,s in enumerate(specs)}
    fixed = {}
    def find(kind,h,m=12,shape=1.,cap=1.):
        keys = [(i,s) for i,s in enumerate(specs) if s.kind==kind and s.width==m and s.shape==shape and s.cap==cap]
        if not keys: raise ValueError("the benchmark grids must include width 12")
        return min(keys, key=lambda pair:abs(pair[1].h-h))[0]
    fixed["NN_expanding"] = find("expanding",0)
    fixed["NN_rolling_short"] = find("uniform",32)
    fixed["NN_rolling_long"] = find("uniform",256)
    fixed["NN_exponential"] = find("exponential",256)
    rf = {}
    logs, ledger, candidate_risks, eval_times = [], [], [], []
    loss_sums, n_losses = {}, 0
    selected_previous, switches = {}, {}
    bank_predictions = np.full((T,K), np.nan, dtype=np.float32)
    selections = []
    times = {"neural_fit":0., "forest_fit":0., "mu_fit":0., "online":0.}
    D, A, radius = np.zeros(K), np.zeros(K), np.ones(K)
    for t in range(start,T):
        is_refit = (t-start) % stride == 0
        if is_refit:
            extra = None
            if detectors:
                h_a, h_p = detectors.windows(t)
                extra = {K:h_a, K+1:h_p}
            tick = time.perf_counter()
            bank.fit(X[:t], y[:t], extra)
            times["neural_fit"] += time.perf_counter()-tick
            cp = bank.predict(X[max(0,t-64):t])[:,:K]
            D,A,radius = comparison_penalties(specs,cp,bank.diag[:K],variance_scale,
                                               config.get("lepski_threshold",.25))
            tick = time.perf_counter()
            for model in mu.values(): model.fit(X[:t],y[:t])
            times["mu_fit"] += time.perf_counter()-tick
            if config.get("random_forest",True):
                tick = time.perf_counter()
                for name,h in [("RF_expanding",t),("RF_rolling",256)]:
                    model = RandomForestRegressor(n_estimators=int(config.get("rf_trees",32)),
                        min_samples_leaf=5, max_features=1., random_state=seed, n_jobs=1)
                    model.fit(X[max(0,t-h):t],y[max(0,t-h):t])
                    rf[name] = model
                times["forest_fit"] += time.perf_counter()-tick
        # Everything in `choice` is chosen BEFORE reading this period's label.
        choice = dict(fixed)
        for name,ids in masks.items(): choice[name] = selectors[name].choose(ids)
        ids = masks["Joint_CV"]
        choice["Joint_Lepski"] = int(ids[np.argmin((D+A+.25*radius)[ids])])
        allowed = ids[D[ids] <= np.min(D[ids]) + .25*radius[ids]]
        choice["Joint_Hybrid"] = selectors["Joint_CV"].choose(allowed)
        if detectors:
            choice["ADWIN_NN"], choice["PageHinkley_NN"] = K,K+1
        mixture_vectors = {name:model.vector() for name,model in mixes.items()}
        for name, j in choice.items():
            if name in selected_previous and selected_previous[name] != j:
                switches[name] = switches.get(name,0)+1
            selected_previous[name] = j
        p_all = bank.predict(X[t:t+1])[0]
        p = p_all[:K]
        bank_predictions[t] = p
        pred = {name:float(p_all[j]) for name,j in choice.items()}
        pred.update({name:float(v@p) for name,v in mixture_vectors.items()})
        rls_pred={name:float(model.predict(X[t:t+1])[0]) for name,model in rls.items()}
        pred.update(rls_pred)
        recent=float(np.clip(np.mean(y[max(0,t-64):t]),-4.,4.))
        simple=np.array([0.,recent,rls_pred["RLS_099"],rls_pred["RLS_0995"]])
        ext=ext_pool.vector(p,simple)
        advanced_vectors={
            "AdaHedge_NN":ada_nn.vector(),
            "ShareGrid_NN":share_nn.vector(),
            "AdaHedge_Extended":ada_ext.vector(),
            "AdaHedge_ComplexityPrior":ada_prior.vector(),
            "ShareGrid_Extended":share_ext.vector(),
            "ShareGrid_ComplexityPrior":share_prior.vector(),
        }
        pred["AdaHedge_NN"]=float(advanced_vectors["AdaHedge_NN"]@p)
        pred["ShareGrid_NN"]=float(advanced_vectors["ShareGrid_NN"]@p)
        for name in ("AdaHedge_Extended","AdaHedge_ComplexityPrior","ShareGrid_Extended","ShareGrid_ComplexityPrior"):
            pred[name]=float(advanced_vectors[name]@ext)
        pred["ShareGrid_Extended_Top3"]=float(sparse_vector(advanced_vectors["ShareGrid_Extended"],3)@ext)
        pred["ShareGrid_Extended_Top5"]=float(sparse_vector(advanced_vectors["ShareGrid_Extended"],5)@ext)
        pred.update({name:float(model.predict(X[t:t+1])[0]) for name,model in mu.items()})
        pred.update({name:float(model.predict(X[t:t+1])[0]) for name,model in rf.items()})
        if online:
            pred.update({name:float(v[0]) for name,v in online.predict(X[t:t+1]).items()})
        pred["Mean_recent"] = recent
        pred["Zero"] = 0.
        if is_refit and t >= burn and nprobe:
            xp, truth = generator.probes(t, nprobe, 0)
            pp_all = bank.predict(xp)
            pp = pp_all[:,:K]
            risk_per_expert = np.mean((pp-truth[:,None])**2, axis=0)
            candidate_risks.append(risk_per_expert)
            eval_times.append(t)
            evaluated = {name:pp_all[:,j] for name,j in choice.items()}
            evaluated.update({name:pp@v for name,v in mixture_vectors.items()})
            rls_probe={name:model.predict(xp) for name,model in rls.items()}
            evaluated.update(rls_probe)
            simple_probe=np.column_stack([np.zeros(nprobe),np.full(nprobe,recent),
                                          rls_probe["RLS_099"],rls_probe["RLS_0995"]])
            ext_probe=ext_pool.matrix(pp,simple_probe)
            evaluated["AdaHedge_NN"]=pp@advanced_vectors["AdaHedge_NN"]
            evaluated["ShareGrid_NN"]=pp@advanced_vectors["ShareGrid_NN"]
            for name in ("AdaHedge_Extended","AdaHedge_ComplexityPrior","ShareGrid_Extended","ShareGrid_ComplexityPrior"):
                evaluated[name]=ext_probe@advanced_vectors[name]
            evaluated["ShareGrid_Extended_Top3"]=ext_probe@sparse_vector(advanced_vectors["ShareGrid_Extended"],3)
            evaluated["ShareGrid_Extended_Top5"]=ext_probe@sparse_vector(advanced_vectors["ShareGrid_Extended"],5)
            evaluated.update({name:model.predict(xp) for name,model in mu.items()})
            evaluated.update({name:model.predict(xp) for name,model in rf.items()})
            if online: evaluated.update(online.predict(xp))
            evaluated["Mean_recent"] = np.full(nprobe,pred["Mean_recent"])
            evaluated["Zero"] = np.zeros(nprobe)
            # Independent oracle-selection probes. Never passed to any feasible rule.
            xo, fo = generator.probes(t,int(config.get("oracle_probes",512)),1)
            po = bank.predict(xo)[:,:K]
            oracle_idx = int(np.argmin(np.mean((po-fo[:,None])**2,axis=0)))
            evaluated["Oracle_grid"] = pp[:,oracle_idx]
            c_eval = {**choice,"Oracle_grid":oracle_idx}
            last_break = max([v for v in generator.breakpoints if v <= t], default=-10**8)
            after_break = 0 <= t-last_break < 64
            for name,values in evaluated.items():
                j = c_eval.get(name)
                info = bank.diag[j] if j is not None else {}
                sp = specs_all[j] if j is not None else None
                md=ext_pool.diagnostics(advanced_vectors[name]) if name in advanced_vectors and len(advanced_vectors[name])==ext_pool.size else {}
                logs.append({"scenario":scenario,"seed":seed,"t":t,"method":name,
                    "excess_risk":float(np.mean((np.asarray(values)-truth)**2)),
                    "post_break_64":int(after_break),
                    "width":sp.width if sp else np.nan,"h":sp.h if sp else np.nan,
                    "shape":sp.shape if sp else np.nan,"kernel":sp.kind if sp else "mixture_or_external",
                    "neff":info.get("neff",np.nan),"support":info.get("support",np.nan),
                    "v_over_q":info.get("v_over_q",np.nan),
                    "selected":j if j is not None else -1,
                    "meta_effective_experts":md.get("meta_effective_experts",np.nan),
                    "meta_zero_mass":md.get("meta_zero_mass",np.nan),
                    "meta_neural_mass":md.get("meta_neural_mass",np.nan),
                    "meta_top_weight":md.get("meta_top_weight",np.nan)})
        selections.append({"t":t,**{k:int(v) for k,v in choice.items()}})
        # The response is used only below: score, detect, then update online learners.
        yy = float(y[t])
        if t >= burn:
            n_losses += 1
            for name,value in pred.items(): loss_sums[name] = loss_sums.get(name,0.)+(value-yy)**2
        for selector in selectors.values(): selector.update(p,yy)
        for mix in mixes.values(): mix.update(p,yy)
        ada_nn.update(p,yy,meta_scale); share_nn.update(p,yy)
        ada_ext.update(ext,yy,meta_scale); ada_prior.update(ext,yy,meta_scale)
        share_ext.update(ext,yy); share_prior.update(ext,yy)
        if detectors:
            detectors.update(t,(pred["ADWIN_NN"]-yy)**2,(pred["PageHinkley_NN"]-yy)**2,score_scale)
        tick = time.perf_counter()
        for model in rls.values(): model.update(X[t],yy)
        if online: online.update(X[t],yy)
        times["online"] += time.perf_counter()-tick
        ledger.append({"t":t,**pred})
    df = pd.DataFrame(logs)
    summaries = []
    for method, rows in (df.groupby("method") if len(df) else []):
        post = rows.loc[rows.post_break_64==1,"excess_risk"]
        summaries.append({"scenario":scenario,"seed":seed,"method":method,
            "mean_risk":rows.excess_risk.mean(),"median_risk":rows.excess_risk.median(),
            "post_break_risk":post.mean() if len(post) else np.nan,
            "prequential_mse":loss_sums.get(method,np.nan)/max(1,n_losses),
            "mean_width":rows.width.mean(),"mean_neff":rows.neff.mean(),
            "switches":switches.get(method,0),"evaluations":len(rows)})
    out = Path(out_dir)/f"{scenario}__{seed}"
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"trajectory.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"summary.csv",index=False)
    pd.DataFrame(ledger).to_csv(out/"prequential.csv",index=False)
    pd.DataFrame(selections).to_csv(out/"selections.csv",index=False)
    np.savez_compressed(out/"candidate_ledger.npz", predictions=bank_predictions,
                        y=y, eval_times=np.array(eval_times),risk=np.array(candidate_risks))
    meta = {"scenario":scenario,"seed":seed,"T":T,"d":d,"candidate_count":K,"extended_expert_count":ext_pool.size,
        "neural_fits":bank.nfits,"evaluation_times":eval_times,"breakpoints":generator.breakpoints,
        "stress":STRESS.get(scenario),"seconds":time.perf_counter()-started,"component_seconds":times,
        "detector_alarms":detectors.alarms if detectors else {},
        "config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        "status":"complete"}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--jobs",type=int,default=1)
    ap.add_argument("--resume",action="store_true")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    root = Path(__file__).parent
    sources = {p.name:file_hash(p) for p in root.glob("*.py")}
    manifest = {"config":config,"source_sha256":sources,"python":platform.python_version(),
        "numpy":np.__version__,"torch":torch.__version__,"scipy":scipy.__version__,
        "platform":platform.platform(),"workers":args.jobs,"completed":[],"failures":[]}
    if config.get("river",True):
        import river
        manifest["river"] = river.__version__
    start = time.perf_counter()
    jobs = [(s,int(seed)) for s in config.get("scenarios",SCENARIOS) for seed in config["seeds"]]
    to_run = []
    for scenario,seed in jobs:
        p = out/f"{scenario}__{seed}"/"metadata.json"
        if args.resume and p.exists():
            old = json.loads(p.read_text())
            expected = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
            if old.get("status")=="complete" and old.get("config_hash")==expected:
                manifest["completed"].append(old); continue
        to_run.append((scenario,seed))
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(run_stream,config,s,seed,str(out)):(s,seed) for s,seed in to_run}
        for future in as_completed(futures):
            s,seed = futures[future]
            try:
                meta = future.result()
                manifest["completed"].append(meta)
                print(f"{len(manifest['completed'])}/{len(jobs)} {s} seed={seed} {meta['seconds']:.1f}s",flush=True)
            except Exception:
                manifest["failures"].append({"scenario":s,"seed":seed,"traceback":traceback.format_exc()})
                print(f"FAILED {s} {seed}\n{manifest['failures'][-1]['traceback']}",flush=True)
            manifest["wall_seconds"] = time.perf_counter()-start
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    manifest["status"] = "complete" if not manifest["failures"] and len(manifest["completed"])==len(jobs) else "incomplete"
    manifest["wall_seconds"] = time.perf_counter()-start
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if manifest["status"] != "complete":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
