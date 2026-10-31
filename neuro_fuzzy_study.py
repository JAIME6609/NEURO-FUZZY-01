#!/usr/bin/env python3
"""Reproducible neuro-fuzzy experiments for chronic-disease decision support.

All records are simulated. This program does not diagnose disease, validate a
medical device, or reproduce patient cohorts from the cited clinical studies.

Run: python neuro_fuzzy_study.py --output results
Install missing dependencies automatically only when explicitly requested:
     python neuro_fuzzy_study.py --install-missing --output results
The default run uses data seed 20261025 and split seeds 11, 29, and 47.
Tables, figures, predictions, model parameters and audit files are placed under
5.1, 5.2 and 5.3. No external data download or network service is required.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import hashlib
import zipfile

DEPENDENCIES = {'numpy':'numpy>=1.24', 'scipy':'scipy>=1.10',
                'pandas':'pandas>=2.0', 'sklearn':'scikit-learn>=1.3',
                'matplotlib':'matplotlib>=3.7', 'threadpoolctl':'threadpoolctl>=3.1'}

def ensure_dependencies(install=False):
    """Check imports; print a directly executable installation command on error."""
    missing = [spec for name, spec in DEPENDENCIES.items()
               if importlib.util.find_spec(name) is None]
    if missing:
        command = [sys.executable, '-m', 'pip', 'install', *missing]
        if not install:
            raise SystemExit('Missing packages. Run:\n' + subprocess.list2cmdline(command)
                             + '\nOr rerun with --install-missing.')
        subprocess.run(command, check=True)
    for name in DEPENDENCIES:
        importlib.import_module(name)


def initialize_numeric_libraries():
    global np, pd, plt, minimize, expit, logsumexp, KMeans, StandardScaler
    global LogisticRegression, Ridge, MLPRegressor, MLPClassifier
    global roc_auc_score, brier_score_loss, roc_curve, mean_absolute_error
    global mean_squared_error, calibration_curve, threadpool_limits
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.metrics import (roc_auc_score, brier_score_loss, roc_curve,
                                 mean_absolute_error, mean_squared_error)
    from sklearn.calibration import calibration_curve
    from threadpoolctl import threadpool_limits
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.dpi':240, 'figure.constrained_layout.use':True})


class NeuroFuzzyRegressor:
    """First-order Gaussian TSK model with profiled ridge consequents.

    Membership centers and log-widths are optimized by bounded L-BFGS-B.
    Every objective evaluation solves the ridge normal equations for the
    consequent coefficients. The analytic gradient uses the envelope theorem;
    it is the gradient of the profiled objective, not a frozen-coefficient
    approximation to a different loss. All calculations use float64.
    """
    def __init__(self, rules=4, ridge=0.001, max_iter=1000, seed=11):
        self.rules, self.ridge, self.max_iter, self.seed = rules, ridge, max_iter, seed

    def _weights(self, x, centers=None, logs=None):
        c = self.centers_ if centers is None else centers
        l = self.log_widths_ if logs is None else logs
        delta = x[:,None,:] - c[None,:,:]
        log_firing = -.5 * np.sum(delta**2 * np.exp(-2*l)[None,:,:], axis=2)
        weights = np.exp(log_firing-logsumexp(log_firing,axis=1,keepdims=True))
        return weights, delta, log_firing

    def _profile(self, vector, x, y, return_details=False):
        d=x.shape[1]; k=self.rules
        centers=vector[:k*d].reshape(k,d); logs=vector[k*d:].reshape(k,d)
        weights,delta,_=self._weights(x,centers,logs)
        augmented=np.column_stack([np.ones(len(x)),x])
        design=(weights[:,:,None]*augmented[:,None,:]).reshape(len(x),-1)
        penalty=len(x)*self.ridge
        theta=np.linalg.solve(design.T@design+penalty*np.eye(design.shape[1]),design.T@y)
        predictions=design@theta; residual=predictions-y
        objective=np.mean(residual**2)+self.ridge*np.sum(theta**2)
        local=augmented@theta.reshape(k,d+1).T
        score_gradient=2*residual[:,None]/len(x)*weights*(local-predictions[:,None])
        invvar=np.exp(-2*logs)
        center_gradient=np.einsum('nk,nkd,kd->kd',score_gradient,delta,invvar)
        width_gradient=np.einsum('nk,nkd,kd->kd',score_gradient,delta**2,invvar)
        gradient=np.r_[center_gradient.ravel(),width_gradient.ravel()]
        if return_details: return objective, gradient, theta
        return objective,gradient

    def fit(self,x,y):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
        if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Fit requires aligned finite two-dimensional features and targets.')
        if len(x)<self.rules or self.ridge<=0: raise ValueError('Invalid rule count or ridge penalty.')
        initial=KMeans(n_clusters=self.rules,n_init=10,random_state=self.seed).fit(x).cluster_centers_
        v=np.r_[initial.ravel(), np.zeros(initial.size)]
        bounds=[(-4.,4.)]*initial.size+[(np.log(.35),np.log(3.))]*initial.size
        fit=minimize(self._profile,v,args=(x,y),jac=True,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':self.max_iter,'ftol':1e-10,'gtol':1e-6})
        self.centers_=fit.x[:initial.size].reshape(initial.shape)
        self.log_widths_=fit.x[initial.size:].reshape(initial.shape)
        self.coefficients_=self._profile(fit.x,x,y,True)[2].reshape(self.rules,x.shape[1]+1)
        self.optimization_={'success':bool(fit.success),'status':int(fit.status),
                            'message':str(fit.message),'iterations':int(fit.nit),
                            'objective':float(fit.fun),'gradient_norm':float(np.linalg.norm(fit.jac))}
        if not np.isfinite(self.coefficients_).all(): raise FloatingPointError('Nonfinite coefficients.')
        return self

    def predict(self,x):
        weights,_,_=self._weights(np.asarray(x,dtype=float))
        local=np.column_stack([np.ones(len(x)),x])@self.coefficients_.T
        return np.sum(weights*local,axis=1)

    def audit(self,x):
        weights,_,logs=self._weights(x)
        entropy=-np.sum(weights*np.log(np.maximum(weights,1e-300)),axis=1)
        return {'weights':weights,'entropy':entropy,'support':np.max(logs,axis=1),
                'effective_rules':np.exp(entropy)}


def make_data(seed):
    """Generate four explicitly artificial mechanisms with observable noisy inputs.

    Labels use latent clean values. Stored inputs include measurement error.
    Renal/lung probabilities are simulation laws, never clinical risk equations.
    Temperature is in degrees Celsius; pressure is in mmHg; pulse transit time
    is in milliseconds. Albuminuria is log(1 + mg/g), dimensionless.
    """
    r=np.random.default_rng(seed); data={}
    n=2400
    age=r.uniform(35,85,n); egfr=r.uniform(20,100,n); albumin=r.uniform(0,7,n)
    sbp=np.clip(r.normal(135,18,n),90,195); diabetes=r.binomial(1,.35,n)
    logit=-2.6+.04*(age-60)+.055*(60-egfr)+.35*albumin+.012*(sbp-130)+.5*diabetes
    logit+=.65*np.tanh((50-egfr)/12)*(albumin-3)/2
    probability=expit(logit); y=r.binomial(1,probability)
    x=np.column_stack([age,egfr+r.normal(0,3,n),albumin+r.normal(0,.15,n),sbp+r.normal(0,3,n),diabetes])
    data['renal']={'x':x,'y':y,'groups':np.arange(n),'names':['age','eGFR','log_albuminuria','systolic_pressure','diabetes'],
                   'kind':'classification','oracle':probability,'audit_group':(age>=65).astype(int)}
    n=2000
    age=r.uniform(40,85,n); diameter=r.uniform(3,25,n); spiculation=r.beta(2,3,n)
    texture=r.beta(2,2,n); smoking=r.uniform(0,65,n)
    logit=-4+.12*diameter+2.2*spiculation+.8*texture+.018*smoking+.02*(age-60)
    logit+=1.4*spiculation*np.tanh((diameter-12)/4)
    probability=expit(logit); y=r.binomial(1,probability)
    x=np.column_stack([age,diameter+r.normal(0,.8,n),np.clip(spiculation+r.normal(0,.04,n),0,1),texture,smoking])
    data['lung']={'x':x,'y':y,'groups':np.arange(n),'names':['age','diameter','spiculation','texture','smoking'],
                  'kind':'classification','oracle':probability,'audit_group':(age>=65).astype(int)}
    persons=200; days=30; group=np.repeat(np.arange(persons),days); day=np.tile(np.arange(days),persons)
    age=np.repeat(r.uniform(35,85,persons),days); stiffness=np.repeat(r.normal(0,1,persons),days)
    activity=r.uniform(0,1,len(group)); phase=np.repeat(r.uniform(0,2*np.pi,persons),days)
    innovations=r.normal(0,2,(persons,days)); disturbance=np.zeros_like(innovations)
    for t in range(days): disturbance[:,t]=innovations[:,t]+(.7*disturbance[:,t-1] if t else 0)
    hr=65+17*activity+5*np.sin(2*np.pi*day/7+phase)+r.normal(0,3,len(group))
    target=120+.45*(age-50)+10*stiffness+12*activity+8*np.sin(2*np.pi*day/14+phase)+disturbance.ravel()
    ptt=260-0.7*(target-100)+.9*(hr-65)+.0015*(target-120)**2+r.normal(0,2,len(group))
    x=np.column_stack([ptt,hr,age,activity])
    data['monitoring']={'x':x,'y':target,'groups':group,'day':day,'names':['pulse_transit_time','heart_rate','age','activity'],
                        'kind':'regression','audit_group':(age>=65).astype(int)}
    n=2400; true=r.uniform(90,180,n); temperature=r.uniform(15,35,n); battery=r.uniform(.2,1,n)
    raw=true+.003*(true-125)**2+.25*(temperature-25)+4*(1-battery)+r.normal(0,1.2,n)
    x=np.column_stack([raw,temperature,battery])
    data['sensor']={'x':x,'y':true,'groups':np.arange(n),'names':['raw_pressure','temperature','battery'],
                    'kind':'regression','audit_group':(battery<.5).astype(int)}
    return data


def split_groups(groups,seed):
    """Assign whole subjects to 60/10/10/20 percent disjoint partitions."""
    unique=np.unique(groups); order=np.random.default_rng(seed).permutation(unique)
    a,b,c=[int(len(order)*q) for q in (.6,.7,.8)]
    partitions=[np.flatnonzero(np.isin(groups,v)) for v in np.split(order,[a,b,c])]
    assert len(set(np.concatenate(partitions)))==len(groups)
    for i in range(4):
        for j in range(i): assert not set(groups[partitions[i]])&set(groups[partitions[j]])
    return partitions


def calibrate(score,y):
    """Positive-slope sigmoid calibration fitted only on calibration subjects."""
    score=np.asarray(score);y=np.asarray(y)
    def objective(v):
        slope=np.exp(v[0]);z=slope*score+v[1];error=expit(z)-y
        return (np.mean(np.logaddexp(0,z)-y*z)+1e-6*np.dot(v,v),
                np.array([np.mean(error*score)*slope,np.mean(error)])+2e-6*v)
    fit=minimize(objective,[0.,0.],jac=True,method='L-BFGS-B',bounds=[(-5,5),(-15,15)])
    return [float(np.exp(fit.x[0])),float(fit.x[1])]


def metrics(y,p,kind):
    if kind=='classification':
        return {'AUROC':roc_auc_score(y,p),'Brier':brier_score_loss(y,p),
                'Sensitivity':float(np.mean(p[y==1]>=.5)), 'Specificity':float(np.mean(p[y==0]<.5))}
    err=p-y
    return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':mean_absolute_error(y,p),
            'Bias':float(np.mean(err)),'SD_error':float(np.std(err,ddof=1))}


def bootstrap(y,p,kind,groups,seed=810,other=None,repetitions=1000):
    """Percentile intervals; resample whole subjects and keep paired predictions."""
    rng=np.random.default_rng(seed);u=np.unique(groups); lookup={g:np.where(groups==g)[0] for g in u}
    name='AUROC' if kind=='classification' else 'RMSE'; vals=[]
    for _ in range(repetitions):
        ids=np.concatenate([lookup[g] for g in rng.choice(u,len(u),replace=True)])
        if kind=='classification' and len(np.unique(y[ids]))<2:continue
        value=metrics(y[ids],p[ids],kind)[name]
        if other is not None:value-=metrics(y[ids],other[ids],kind)[name]
        vals.append(value)
    return [float(v) for v in np.quantile(vals,[.025,.975])]


def fit_task(ds,seed):
    tr,tu,ca,te=split_groups(ds['groups'],seed);x,y=ds['x'],ds['y'];kind=ds['kind']
    scaler=StandardScaler().fit(x[tr]);z=scaler.transform(x)
    target_mean=float(y[tr].mean()) if kind=='regression' else 0.
    target_scale=float(y[tr].std()) if kind=='regression' else 1.
    target=(y-target_mean)/target_scale
    models={};tuning=[];elapsed={}
    start=time.perf_counter();best=None
    for count in (4,8):
        candidate=NeuroFuzzyRegressor(count,seed=seed).fit(z[tr],target[tr])
        loss=float(np.mean((candidate.predict(z[tu])-target[tu])**2))
        tuning.append({'family':'ANFIS','rules':count,'validation_mse':loss,**candidate.optimization_})
        if best is None or loss<best[0]:best=(loss,candidate)
    models['ANFIS']=best[1];elapsed['ANFIS']=time.perf_counter()-start
    start=time.perf_counter();best=None
    for alpha in (.001,.1):
        if kind=='classification':
            candidate=MLPClassifier(hidden_layer_sizes=(24,12),alpha=alpha,solver='lbfgs',max_iter=2000,random_state=seed)
        else:
            candidate=MLPRegressor(hidden_layer_sizes=(24,12),alpha=alpha,solver='lbfgs',max_iter=2000,random_state=seed)
        candidate.fit(z[tr],target[tr])
        pred=candidate.predict_proba(z[tu])[:,1] if kind=='classification' else candidate.predict(z[tu])
        loss=float(np.mean((pred-target[tu])**2));tuning.append({'family':'MLP','alpha':alpha,'validation_mse':loss,'iterations':int(candidate.n_iter_)})
        if best is None or loss<best[0]:best=(loss,candidate)
    models['MLP']=best[1];elapsed['MLP']=time.perf_counter()-start
    start=time.perf_counter()
    linear=LogisticRegression(C=1,max_iter=2000,random_state=seed) if kind=='classification' else Ridge(alpha=1.)
    linear.fit(z[tr],target[tr]);models['Linear']=linear;elapsed['Linear']=time.perf_counter()-start
    calibrators={};predictions={};rows=[]
    def raw(model,zz):
        return model.predict_proba(zz)[:,1] if hasattr(model,'predict_proba') else model.predict(zz)
    for name,model in models.items():
        score=raw(model,z[te])
        if kind=='classification':
            calibrators[name]=calibrate(raw(model,z[ca]),y[ca]);a,b=calibrators[name];p=expit(a*score+b)
        else:p=score*target_scale+target_mean
        predictions[name]=p
        rows.append({'model':name,'split_seed':seed,'test_rows':len(te),**metrics(y[te],p,kind),
                     'fit_seconds':elapsed[name]})
    def predict(model_name,new_x):
        score=raw(models[model_name],scaler.transform(new_x))
        if kind=='classification':
            a,b=calibrators[model_name];return expit(a*score+b)
        return score*target_scale+target_mean
    return {'models':models,'scaler':scaler,'partitions':(tr,tu,ca,te),'predictions':predictions,'rows':rows,
            'predict':predict,'tuning':tuning,'target_mean':target_mean,'target_scale':target_scale,'calibrators':calibrators}


def save_figure(fig,path):
    for extension in ('png', 'pdf'):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, bbox_inches='tight')
        path.with_suffix('.' + extension).write_bytes(buffer.getvalue())
    plt.close(fig)


def run(output,data_seed,split_seeds,bootstrap_repetitions):
    output.mkdir(parents=True,exist_ok=True)
    for sub in ('5.1','5.2','5.3'):(output/sub).mkdir(exist_ok=True)
    data=make_data(data_seed);fits={};all_rows=[];manifest={'data_seed':data_seed,'split_seeds':split_seeds,
        'data_status':'All records are simulated; no patients or clinical images are included.',
        'environment':{'python':sys.version,'platform':platform.platform(),
        'packages':{name:importlib.metadata.version('scikit-learn' if name=='sklearn' else name) for name in DEPENDENCIES}},
        'tasks':{}}
    for task,ds in data.items():
        section='5.1' if ds['kind']=='classification' else '5.2'
        for seed in split_seeds:
            print(f'Fitting {task}, split {seed}',flush=True)
            fit=fit_task(ds,seed)
            for row in fit['rows']:row['task']=task;all_rows.append(row)
            if seed==split_seeds[0]:fits[task]=fit
        fit=fits[task];tr,tu,ca,te=fit['partitions']
        partition=np.empty(len(ds['y']),dtype=object)
        for name,ids in zip(('train','tune','calibration','test'),(tr,tu,ca,te)):partition[ids]=name
        df=pd.DataFrame(ds['x'],columns=ds['names']);df['target']=ds['y'];df['subject_id']=ds['groups'];df['partition']=partition
        if 'day' in ds:df['day']=ds['day']
        df.to_csv(output/section/f'{task}_synthetic_data.csv',index=False)
        pred=pd.DataFrame({'row_id':te,'subject_id':ds['groups'][te],'observed':ds['y'][te],**fit['predictions']})
        pred.to_csv(output/section/f'{task}_test_predictions.csv',index=False)
        m=fit['models']['ANFIS'];details={'features':ds['names'],'centers_standardized':m.centers_.tolist(),
          'widths_standardized':np.exp(m.log_widths_).tolist(),'coefficients':m.coefficients_.tolist(),
          'input_mean':fit['scaler'].mean_.tolist(),'input_scale':fit['scaler'].scale_.tolist(),
          'target_mean':fit['target_mean'],'target_scale':fit['target_scale'],'calibrators':fit['calibrators'],
          'optimizer':m.optimization_,'tuning':fit['tuning']}
        (output/'5.3'/f'{task}_model.json').write_text(json.dumps(details,indent=2))
        manifest['tasks'][task]={'rows':len(df),'features':ds['names'],'partitions':{k:len(ids) for k,ids in zip(('train','tune','calibration','test'),(tr,tu,ca,te))},
                                 'test_subjects':len(np.unique(ds['groups'][te]))}
    all_metrics=pd.DataFrame(all_rows);all_metrics.to_csv(output/'5.3'/'repeated_split_metrics.csv',index=False)
    primary=all_metrics[all_metrics.split_seed==split_seeds[0]].copy(); intervals=[]
    for task,ds in data.items():
        fit=fits[task];te=fit['partitions'][3]
        for name,p in fit['predictions'].items():
            lo,hi=bootstrap(ds['y'][te],p,ds['kind'],ds['groups'][te],other=None,repetitions=bootstrap_repetitions)
            intervals.append({'task':task,'model':name,'CI_low':lo,'CI_high':hi})
    primary=primary.merge(pd.DataFrame(intervals),on=['task','model'])
    primary[primary.task.isin(['renal','lung'])].to_csv(output/'5.1'/'table_1_classification.csv',index=False)
    primary[primary.task.isin(['monitoring','sensor'])].to_csv(output/'5.2'/'table_3_regression.csv',index=False)
    paired=[]
    for task,ds in data.items():
        f=fits[task];te=f['partitions'][3]
        for other in ('MLP','Linear'):
            lo,hi=bootstrap(ds['y'][te],f['predictions']['ANFIS'],ds['kind'],ds['groups'][te],other=f['predictions'][other],repetitions=bootstrap_repetitions)
            metric='AUROC' if ds['kind']=='classification' else 'RMSE'
            diff=metrics(ds['y'][te],f['predictions']['ANFIS'],ds['kind'])[metric]-metrics(ds['y'][te],f['predictions'][other],ds['kind'])[metric]
            paired.append({'task':task,'contrast':f'ANFIS minus {other}','metric':metric,'difference':diff,'CI_low':lo,'CI_high':hi})
    paired=pd.DataFrame(paired);paired.to_csv(output/'5.3'/'paired_differences_all.csv',index=False)
    paired[paired.task.isin(['renal','lung'])].to_csv(output/'5.1'/'table_2_paired_discrimination.csv',index=False)
    colors={'ANFIS':'#173f5f','MLP':'#d17a27','Linear':'#52806b'}
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,task in zip(axes,['renal','lung']):
        f=fits[task];y=data[task]['y'][f['partitions'][3]]
        for name,p in f['predictions'].items():
            a,b,_=roc_curve(y,p);ax.plot(a,b,label=f'{name}  {roc_auc_score(y,p):.3f}',color=colors[name])
        ax.plot([0,1],[0,1],':',color='grey');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{task.title()} simulation');ax.legend(fontsize=8)
    save_figure(fig,output/'5.1'/'figure_1_roc')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,task in zip(axes,['renal','lung']):
        f=fits[task];y=data[task]['y'][f['partitions'][3]]
        for name,p in f['predictions'].items():
            observed,predicted=calibration_curve(y,p,n_bins=6,strategy='quantile');ax.plot(predicted,observed,'o-',color=colors[name],label=name)
        ax.plot([0,1],[0,1],':',color='grey');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed event fraction',title=f'{task.title()} simulation');ax.legend(fontsize=8)
    save_figure(fig,output/'5.1'/'figure_2_calibration')
    ds=data['monitoring'];f=fits['monitoring'];te=f['partitions'][3];y=ds['y'][te];p=f['predictions']['ANFIS'];g=ds['groups'][te]
    chosen=np.unique(g)[0];ids=np.where(g==chosen)[0]
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));ax=axes[0]
    ax.plot(ds['day'][te][ids],y[ids],'o-',ms=3,label='Simulated reference',color='black');ax.plot(ds['day'][te][ids],p[ids],'s-',ms=3,label='ANFIS',color=colors['ANFIS']);ax.axhline(140,color='grey',ls=':');ax.set(xlabel='Day',ylabel='Systolic pressure (mmHg)',title=f'First held-out subject ({chosen})');ax.legend(fontsize=8)
    ax=axes[1];err=p-y;ax.scatter((p+y)/2,err,s=8,alpha=.25,color=colors['ANFIS']);bias=err.mean();sd=err.std(ddof=1)
    for v in [bias,bias-1.96*sd,bias+1.96*sd]:ax.axhline(v,color='grey',ls='--')
    ax.set(xlabel='Mean of prediction and reference (mmHg)',ylabel='Prediction minus reference (mmHg)',title='Repeated-measure agreement display')
    save_figure(fig,output/'5.2'/'figure_3_monitoring')
    alerts=[]
    for name,p in f['predictions'].items():
        truth=y>=140;flag=p>=140
        alerts.append({'model':name,'sensitivity':np.mean(flag[truth]),'specificity':np.mean(~flag[~truth]),
                       'false_alerts_per_100_days':100*np.sum(flag&~truth)/len(y),
                       'missed_events_per_100_days':100*np.sum(~flag&truth)/len(y),'reference_high_days':int(truth.sum())})
    pd.DataFrame(alerts).to_csv(output/'5.2'/'table_4_alerts.csv',index=False)
    ds=data['sensor'];f=fits['sensor'];te=f['partitions'][3];y=ds['y'][te];p=f['predictions']['ANFIS'];raw=ds['x'][te,0]
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));axes[0].scatter(y,raw-y,s=10,alpha=.3,label='Raw sensor',color=colors['MLP']);axes[0].scatter(y,p-y,s=10,alpha=.3,label='ANFIS corrected',color=colors['ANFIS']);axes[0].axhline(0,color='grey',ls=':');axes[0].set(xlabel='Reference pressure (mmHg)',ylabel='Measurement error (mmHg)',title='Synthetic low-cost sensor');axes[0].legend(fontsize=8)
    axes[1].hist(raw-y,bins=25,alpha=.5,label='Raw sensor',color=colors['MLP']);axes[1].hist(p-y,bins=25,alpha=.6,label='ANFIS corrected',color=colors['ANFIS']);axes[1].set(xlabel='Measurement error (mmHg)',ylabel='Records',title='Residual distributions');axes[1].legend(fontsize=8)
    save_figure(fig,output/'5.2'/'figure_4_sensor')
    (output/'5.2'/'sensor_raw_metrics.json').write_text(json.dumps(metrics(y,raw,'regression'),indent=2))
    # Stress tests alter held-out measurements, keeping labels and fitted models fixed.
    stress=[];ds=data['renal'];f=fits['renal'];te=f['partitions'][3];x=ds['x'][te].copy();y=ds['y'][te]
    rng=np.random.default_rng(891);noise=rng.normal(size=x.shape);uniform=rng.random(x.shape)
    scenarios={'Clean':x.copy(),'Noise 0.25 SD':x+.25*noise*f['scaler'].scale_,
               'Noise 0.50 SD':x+.5*noise*f['scaler'].scale_,'Missing 20%':x.copy(),'eGFR bias +10':x.copy()}
    # Diabetes remains binary under noise; missingness may affect any feature.
    for key in ['Noise 0.25 SD','Noise 0.50 SD']:scenarios[key][:,4]=x[:,4]
    medians=np.median(ds['x'][f['partitions'][0]],axis=0)
    scenarios['Missing 20%']=np.where(uniform<.2,medians,x)
    scenarios['eGFR bias +10'][:,1]+=10
    for scenario,new_x in scenarios.items():
        for name in f['models']:
            pred=f['predict'](name,new_x);stress.append({'scenario':scenario,'model':name,**metrics(y,pred,'classification')})
    stress=pd.DataFrame(stress);stress.to_csv(output/'5.3'/'table_5_stress.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for ax,measure in zip(axes,['AUROC','Brier']):
        for name in colors:
            dd=stress[stress.model==name];ax.plot(np.arange(len(dd)),dd[measure],'o-',label=name,color=colors[name])
        ax.set_xticks(np.arange(5),['Clean','Noise\n0.25 SD','Noise\n0.50 SD','20%\nmissing','eGFR\n+10']);ax.set(ylabel=measure,title=f'Renal measurement stress: {measure}');ax.legend(fontsize=8)
    save_figure(fig,output/'5.3'/'figure_5_stress')
    audits=[];subgroups=[];fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for task,ds in data.items():
        f=fits[task];te=f['partitions'][3];tr=f['partitions'][0];m=f['models']['ANFIS'];z=f['scaler'].transform(ds['x'][te]);audit=m.audit(z)
        parameter_count=m.centers_.size*2+m.coefficients_.size
        start=time.perf_counter()
        for _ in range(50):f['predict']('ANFIS',ds['x'][te])
        latency=1e6*(time.perf_counter()-start)/(50*len(te))
        audits.append({'task':task,'rules':m.rules,'parameters':parameter_count,'mean_effective_rules':float(audit['effective_rules'].mean()),
                       'median_max_weight':float(np.median(audit['weights'].max(axis=1))),'microseconds_per_record':latency,
                       'optimizer_success':m.optimization_['success'],'iterations':m.optimization_['iterations']})
        for group_value in (0,1):
            mask=ds['audit_group'][te]==group_value
            subgroups.append({'task':task,'group':group_value,'n':int(mask.sum()),**metrics(ds['y'][te][mask],f['predictions']['ANFIS'][mask],ds['kind'])})
        if task in ['renal','lung']:
            ax=axes[0 if task=='renal' else 1];im=ax.imshow(m.centers_,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto')
            ax.set_xticks(range(len(ds['names'])),ds['names'],rotation=35,ha='right',fontsize=8);ax.set_yticks(range(m.rules),[f'Rule {i+1}' for i in range(m.rules)]);ax.set_title(f'{task.title()} antecedent centers')
        # Exact local reconstruction: rule contributions sum to the raw TSK score.
        sample=z[0:1];weights=m.audit(sample)['weights'][0]
        local=(np.r_[1.,sample[0]]@m.coefficients_.T)
        assert np.allclose(np.sum(weights*local),m.predict(sample)[0],atol=1e-10)
        pd.DataFrame({'rule':np.arange(1,m.rules+1),'normalized_weight':weights,'local_output':local,
                      'contribution':weights*local}).to_csv(output/'5.3'/f'{task}_first_test_rule_trace.csv',index=False)
    fig.colorbar(im,ax=axes,shrink=.8,label='Training-standardized center')
    save_figure(fig,output/'5.3'/'figure_6_rule_centers')
    pd.DataFrame(audits).to_csv(output/'5.3'/'table_6_rule_audit.csv',index=False)
    pd.DataFrame(subgroups).to_csv(output/'5.3'/'subgroup_metrics.csv',index=False)
    # Independent finite-difference check of the central mathematical implementation.
    rng=np.random.default_rng(17);xx=rng.normal(size=(25,3));yy=rng.normal(size=25)
    checker=NeuroFuzzyRegressor(2);v=np.r_[rng.normal(size=6),np.zeros(6)]
    value,grad=checker._profile(v,xx,yy);finite=[]
    for j in range(len(v)):
        delta=np.zeros_like(v);delta[j]=1e-6
        finite.append((checker._profile(v+delta,xx,yy)[0]-checker._profile(v-delta,xx,yy)[0])/2e-6)
    error=float(np.max(np.abs(grad-np.array(finite))))
    if error>1e-5:raise AssertionError(f'Analytic gradient failed: {error}')
    manifest['validation']={'maximum_gradient_absolute_error':error,'group_disjointness':'passed',
                             'rule_reconstruction':'passed','bootstrap_repetitions':bootstrap_repetitions,
                             'interval_scope':'Conditional on fitted models; no training-data resampling.'}
    (output/'5.3'/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'README.txt').write_text('All records are simulated. No clinical efficacy or regulatory compliance is established.\n'
        '5.1: renal and lung-descriptor classification; 5.2: repeated monitoring and sensor calibration;\n'
        '5.3: stress, rule audit, model parameters, repeated splits, subgroup diagnostics and manifest.\n'
        'Primary results use the first split seed. Other splits describe sensitivity, not independent replications.\n'
        'CSV files retain unrounded values. PNG and PDF figures contain identical plotted data.\n'
        'The pressure alert threshold is an illustrative analytical threshold, not a treatment recommendation.\n')
    render_final_figures(output)
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'SHA256.json'}
    (output/'SHA256.json').write_text(json.dumps(files,indent=2))
    print(primary.to_string(index=False),flush=True)
    return output



def render_final_figures(output):
    """Rebuild publication figures from saved tables and predictions, without refitting."""
    from sklearn.metrics import roc_curve,roc_auc_score
    from sklearn.calibration import calibration_curve
    import json
    def read_csv(path):
        return pd.read_csv(path, float_precision='round_trip')
    R=output;colors={'ANFIS':'#173f5f','MLP':'#d17a27','Linear':'#52806b'}
    plt.rcParams.update({'font.size':11,'axes.titlesize':11,'legend.fontsize':10})
    def save(fig,rel):
     for ext in ['png','pdf']:
      b=io.BytesIO();fig.savefig(b,format=ext,bbox_inches='tight',dpi=260)
      (R/(rel+'.'+ext)).write_bytes(b.getvalue())
     plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,task in zip(axes,['renal','lung']):
     df=read_csv(R/f'5.1/{task}_test_predictions.csv')
     for name in colors:
      a,b,_=roc_curve(df.observed,df[name]);ax.plot(a,b,label=f'{name}  {roc_auc_score(df.observed,df[name]):.3f}',color=colors[name])
     ax.plot([0,1],[0,1],':',color='grey');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{task.title()} simulation');ax.legend(fontsize=9)
    save(fig,'5.1/figure_1_roc')
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,task in zip(axes,['renal','lung']):
     df=read_csv(R/f'5.1/{task}_test_predictions.csv')
     for name in colors:
      obs,pred=calibration_curve(df.observed,df[name],n_bins=6,strategy='quantile');ax.plot(pred,obs,'o-',label=name,color=colors[name])
     ax.plot([0,1],[0,1],':',color='grey');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed event fraction',title=f'{task.title()} simulation');ax.legend(fontsize=9)
    save(fig,'5.1/figure_2_calibration')
    df=read_csv(R/'5.2/monitoring_test_predictions.csv');raw=read_csv(R/'5.2/monitoring_synthetic_data.csv');ids=df.subject_id==df.subject_id.min();y=df.observed.to_numpy();p=df.ANFIS.to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(8,3.5));ax=axes[0]
    x=raw.loc[df.loc[ids,'row_id'],'day'];ax.plot(x,y[ids],'o-',ms=3,label='Reference',color='black');ax.plot(x,p[ids],'s-',ms=3,label='ANFIS',color=colors['ANFIS']);ax.axhline(140,color='grey',ls=':');ax.set(xlabel='Day',ylabel='Systolic pressure (mmHg)',title=f'First test subject ({df.subject_id.min()})');ax.legend(fontsize=9)
    ax=axes[1];err=p-y;ax.scatter((p+y)/2,err,s=8,alpha=.25,color=colors['ANFIS'])
    for v in [err.mean(),err.mean()-1.96*err.std(ddof=1),err.mean()+1.96*err.std(ddof=1)]:ax.axhline(v,color='grey',ls='--')
    ax.set(xlabel='Mean pressure (mmHg)',ylabel='Prediction error (mmHg)',title='Repeated observations')
    save(fig,'5.2/figure_3_monitoring')
    df=read_csv(R/'5.2/sensor_test_predictions.csv');raw=read_csv(R/'5.2/sensor_synthetic_data.csv');y=df.observed.to_numpy();p=df.ANFIS.to_numpy();rr=raw.loc[df.row_id,'raw_pressure'].to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(8,3.5));ax=axes[0];ax.scatter(y,rr-y,s=10,alpha=.3,label='Raw sensor',color=colors['MLP']);ax.scatter(y,p-y,s=10,alpha=.3,label='ANFIS corrected',color=colors['ANFIS']);ax.axhline(0,color='grey',ls=':');ax.set(xlabel='Reference pressure (mmHg)',ylabel='Measurement error (mmHg)',title='Simulated sensor');ax.legend(fontsize=9)
    ax=axes[1];ax.hist(rr-y,bins=25,alpha=.5,label='Raw sensor',color=colors['MLP']);ax.hist(p-y,bins=25,alpha=.6,label='ANFIS corrected',color=colors['ANFIS']);ax.set(xlabel='Measurement error (mmHg)',ylabel='Records',title='Residual distributions');ax.legend(fontsize=9)
    save(fig,'5.2/figure_4_sensor')
    stress=read_csv(R/'5.3/table_5_stress.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,measure in zip(axes,['AUROC','Brier']):
     for name in colors:
      dd=stress[stress.model==name];ax.plot(np.arange(len(dd)),dd[measure],'o-',label=name,color=colors[name])
     ax.set_xticks(np.arange(5),['Clean','Noise\n0.25 SD','Noise\n0.50 SD','20%\nmissing','eGFR\n+10'],fontsize=9);ax.set(ylabel=measure,title=f'Renal measurement stress');ax.legend(fontsize=9)
    save(fig,'5.3/figure_5_stress')
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    short={'renal':['Age','eGFR','Log ACR','Pressure','Diabetes'],'lung':['Age','Diameter','Spiculation','Texture','Smoking']}
    for ax,task in zip(axes,['renal','lung']):
     m=json.load(open(R/f'5.3/{task}_model.json'));c=np.array(m['centers_standardized']);im=ax.imshow(c,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto');ax.set_xticks(range(c.shape[1]),short[task],rotation=35,ha='right',fontsize=10);ax.set_yticks(range(len(c)),[f'Rule {i+1}' for i in range(len(c))],fontsize=10);ax.set_title(f'{task.title()} rule centers')
    fig.colorbar(im,ax=axes,shrink=.8,label='Standardized center')
    save(fig,'5.3/figure_6_rule_centers')

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--data-seed',type=int,default=20261025)
    parser.add_argument('--split-seeds',type=int,nargs='+',default=[11,29,47])
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--install-missing',action='store_true')
    parser.add_argument('--zip',action='store_true',help='Also create an archive beside the output directory.')
    args=parser.parse_args()
    if args.bootstrap<100 or len(set(args.split_seeds))!=len(args.split_seeds):
        parser.error('Use at least 100 bootstrap replicates and distinct split seeds.')
    ensure_dependencies(args.install_missing);initialize_numeric_libraries()
    # One numerical thread avoids oversubscription and improves repeatability.
    with threadpool_limits(limits=1):run(args.output,args.data_seed,args.split_seeds,args.bootstrap)
    if args.zip:
        with zipfile.ZipFile(args.output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(args.output.rglob('*')):
                if file.is_file():archive.write(file,str(file.relative_to(args.output)))
    print('Completed. All numerical findings are simulation results.',flush=True)

if __name__=='__main__':
    main()
