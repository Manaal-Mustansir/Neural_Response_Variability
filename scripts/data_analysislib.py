import numpy as np
# from lmfit import Model

def outlier(x,median,mad,crit=2.5):
    import numpy as np
    return (x>median+crit*mad) | (x<median-crit*mad)
    
def PSTH_meanmatch_twopopulations(pop_mean1,pop_mean2,pop_var1,pop_var2,count_bins,nboots):
    import numpy as np
    if pop_mean1.shape[1] != pop_mean2.shape[1]:
        raise ValueError('pop_mean1 and pop_mean2 must have the same number of time points')

    fano1 = np.nan * np.ones((pop_mean1.shape[1]))
    fano2 = np.nan * np.ones((pop_mean2.shape[1]))
    mean1 = np.nan * np.ones((pop_mean1.shape[1]))
    mean2 = np.nan * np.ones((pop_mean2.shape[1]))

    # loop time
    for i in range(pop_mean1.shape[1]):
        slope1,slope2,meantmp1,meantmp2 = mean_match(pop_mean1[:,i],pop_mean2[:,i],pop_var1[:,i],pop_var2[:,i],count_bins,nboots)
        fano1[i]  = np.mean(slope1)
        fano2[i]  = np.mean(slope2)

        mean1[i] = np.mean(meantmp1)
        mean2[i] = np.mean(meantmp2)

    return fano1, fano2, mean1, mean2


def PSTH_meanmatch(pop_mean,pop_var,first_bin,last_bin,count_bins,nboots):
    import numpy as np
    pop_mean = pop_mean[:,first_bin:last_bin+1]
    pop_var  = pop_var[:,first_bin:last_bin+1]

    fano  = np.nan * np.ones((pop_mean.shape[1]))
    means = np.nan * np.ones((pop_mean.shape[1]))
    
    # loop time
    for i in range(pop_mean.shape[1]-1):
        slope1,slope2,meantmp1,meantmp2 = mean_match(pop_mean[:,0],pop_mean[:,i+1],pop_var[:,0],pop_var[:,i+1],count_bins,nboots)
        fano[i]  = np.mean(slope2)
        means[i] = np.mean(meantmp2)

    return fano, means
        
def linfit_popFanoPSTH(pop_mean,pop_var,CI_value):
    import statsmodels.api as sm
    #from statsmodels.formula.api import ols
    import matplotlib.pyplot as plt
    
    fano = np.nan * np.ones(pop_mean.shape[1])
    fano_LB = np.nan * np.ones(pop_mean.shape[1])
    fano_UB = np.nan * np.ones(pop_mean.shape[1])
    dd   = np.nan * np.ones((pop_mean.shape[1],2))
    
    # loop time
    for i in range(pop_mean.shape[1]):
        model   = sm.OLS(pop_var[:,i], pop_mean[:,i])
        results = model.fit()
        fano[i] = results.params[0]
        # CIs
        CI      = results.conf_int(CI_value)
        fano_LB[i] = CI[0,0]
        fano_UB[i] = CI[0,1]

    return fano, fano_LB, fano_UB

def select_data(spkC, baseline, thr_spikes):
    spkC_mean = np.mean(spkC,axis=1)
    baseline_mean = np.mean(baseline)
    
    return (np.max(spkC_mean) - baseline_mean) > thr_spikes

def signi_fano(fano1, fano2,fano_boot,crit):
    import numpy.linalg as linalg

    # compute euclidean distance
    D = np.abs(fano1 - fano2)
    return D > np.percentile(np.abs(fano_boot),crit)

def geomean(response1, response2):

    return np.sqrt(response1*response2)


def boot_diff(boot_dist1, boot_dist2,diff,q=95):
    common_mean = np.mean((np.mean(boot_dist1), np.mean(boot_dist2)))
    boot_dist1 = boot_dist1 - np.mean(boot_dist1) + common_mean
    boot_dist2 = boot_dist2 - np.mean(boot_dist2) + common_mean

    diff_dist = boot_dist1 - boot_dist2

    return np.abs(diff) >= np.percentile(diff_dist,q)
    
def signi_laser_fano(fano, fano_L,fano_boot,n_boot):
    import numpy.linalg as linalg

    # compute euclidean distance
    D = linalg.norm(fano-fano_L)
    D_boot = np.zeros(int(n_boot))

    # bootstrap distance assuming the laser had no effect
    boot_inds  = np.random.choice(fano_boot.shape[1],(int(n_boot),fano_boot.shape[1]))
    for i,b_inds in enumerate(boot_inds):
        D_boot[i] = linalg.norm(fano_boot[:,b_inds[0]]-fano_boot[:,b_inds[1]])

    return D > np.percentile(D_boot,97.5)

def mean_match(mean1,mean2,variance1,variance2,count_bins,nboots):
    import statsmodels.api as sm
    # the algorithm brakes with nans
    variance1 = variance1[~np.isnan(mean1)]
    mean1 = mean1[~np.isnan(mean1)]
    variance2 = variance2[~np.isnan(mean2)]
    mean2 = mean2[~np.isnan(mean2)]
    
    bin_mems1 = np.digitize(mean1,count_bins)
    bin_mems2 = np.digitize(mean2,count_bins)
    slope1 = np.empty(0)
    slope2 = np.empty(0)
    for boot in range(nboots):
        inds2remove1 = np.empty(0)
        inds2remove2 = np.empty(0)
        # arange uses half open interval [start,stop)
        for i in np.arange(np.min(count_bins),np.max(count_bins)+1,1):
            n1 = np.sum(bin_mems1 == i)
            n2 = np.sum(bin_mems2 == i)

            if n1 > n2:
                inds = np.where(bin_mems1 == i)[0]
                inds2remove1 = np.append(inds2remove1,np.random.choice(inds,n1-n2,replace=False))
            elif n2 > n1:
                inds = np.where(bin_mems2 == i)[0]
                inds2remove2 = np.append(inds2remove2,np.random.choice(inds,n2-n1,replace=False))
                                     
        meantmp1 = np.delete(mean1,inds2remove1)
        varstmp1 = np.delete(variance1,inds2remove1)
        meantmp2 = np.delete(mean2,inds2remove2)
        varstmp2 = np.delete(variance2,inds2remove2)

        model_1  = sm.OLS(varstmp1, meantmp1)
        results1 = model_1.fit()
        model_2  = sm.OLS(varstmp2, meantmp2)
        results2 = model_2.fit()

        slope1 = np.append(slope1,results1.params[0])
        slope2 = np.append(slope2,results2.params[0])
        
    return slope1,slope2,meantmp1,meantmp2

def saturation_point(response, diams,criteria=1.05):
    m_ind = np.argmax(response)
    inds = np.where(response[m_ind:-1] >= response[-1] * criteria)[0]
    if inds.size == 0:
        surr = diams[-1]
        ret_ind = response.shape[0]
    else:
        surr =  diams[inds[-1]+m_ind]
        ret_ind = inds[-1]+m_ind
        
    return (surr, ret_ind)

def significant_laser(fano_boot_1,fano_boot_2,fano_real1,fano_real2):
    mean_fano_boot_1 = np.mean(fano_boot_1,axis=1)
    mean_fano_boot_2 = np.mean(fano_boot_2,axis=1)
    common_fano_mean = (mean_fano_boot_1 + mean_fano_boot_2) / 2
    diffs = np.zeros(fano_real1.shape[0])

    for i in range(mean_fano_boot_1.shape[0]):
        fano_boot_1[i,:] = fano_boot_1[i,:] - mean_fano_boot_1[i] + common_fano_mean[i]
        fano_boot_2[i,:] = fano_boot_2[i,:] - mean_fano_boot_2[i] + common_fano_mean[i]
        boot_diff = np.abs(fano_boot_1[i,:] - fano_boot_2[i,:])
        p95 = np.percentile(boot_diff,100-(100*np.sqrt(0.05)))
        
        if np.abs(fano_real1[i] - fano_real2[i]) > p95:
            diffs[i] = 1

    return diffs == 1

def z_score(vect):
    import numpy as np
    eps = np.finfo(float).eps
    z_vect = (vect - np.mean(vect)) / (eps + np.std(vect))
    return z_vect
    
def meanvar_PSTH(data,count_window=100,style='same',return_bootdstrs=False,nboots=1000):
    data = data > 0
    if style == 'valid':
        mean_timevector = np.nan*np.ones((data.shape[1] - count_window + 1))
        vari_timevector = np.nan*np.ones((data.shape[1] - count_window + 1))
        tmp  = np.ones((data.shape[0], data.shape[1] - count_window + 1))
    else:
        mean_timevector = np.nan*np.ones((data.shape[1]))
        vari_timevector = np.nan*np.ones((data.shape[1]))
        tmp  = np.ones((data.shape[0], data.shape[1]))
            
    for i in range(data.shape[0]):
        # compute spike counts in sliding window
        tmp[i,:] = np.convolve(data[i,:],np.ones(count_window,),style)
            
    vari_timevector = np.var(tmp,axis=0)
    mean_timevector = np.mean(tmp, axis=0)
    
    if return_bootdstrs:
        boot_inds = np.random.choice(tmp.shape[0],(tmp.shape[0],nboots))
        mean_timevector_booted = np.nan * np.ones((nboots,tmp.shape[1]))
        vari_timevector_booted = np.nan * np.ones((nboots,tmp.shape[1]))
        for i in range(nboots):
            mean_timevector_booted[i,:] = np.mean(tmp[boot_inds[:,i],:],axis=0)
            vari_timevector_booted[i,:] = np.var(tmp[boot_inds[:,i],:],axis=0)

    else:
        mean_timevector_booted = np.array([])
        vari_timevector_booted = np.array([])
            
    #
    return mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted

def fano_PSTH(data,bin_width, boot_num, style='same'):
    eps = np.finfo(float).eps
    # bootstrap fano
    boot_inds = np.random.choice(data.shape[0],(boot_num, data.shape[0]))
    if style == 'valid':
        fano = np.nan*np.ones((boot_num, data.shape[1] - bin_width + 1))
    else:
        fano = np.nan*np.ones((boot_num, data.shape[1]))
        
        # the same exlanation for scaling as above
    for iter, b_inds in enumerate(boot_inds):
        spkC = data[b_inds,:]
        if style == 'valid':
            tmp  = np.ones((spkC.shape[0], spkC.shape[1] - bin_width + 1))
        else:
            tmp  = np.ones((spkC.shape[0], spkC.shape[1]))
            
        for i in range(spkC.shape[0]):
            tmp[i,:] = np.convolve(spkC[i,:],np.ones(bin_width,),style)
            
        fano[iter,:] = (np.var(tmp,axis=0))/ (np.mean(tmp, axis=0) + eps)

    #
    return fano


def fano_binwidths(data, bins):
    eps = np.finfo(float).eps
    # return fano factor for data, over trial, at bins binwidths
    fano = np.zeros(bins.shape[0])

    for b in range(bins.shape[0]):
        for tr in range(data.shape[0]):
            if tr == 0:
                tmp_cn = np.convolve(data[tr,:],np.ones((bins[b],)),'valid')
                convo_bins = np.zeros((data.shape[0],tmp_cn[0:-1:bins[b]].shape[0]))
                convo_bins[tr,:] = tmp_cn[0:-1:bins[b]]

            #
            tmp_cn = np.convolve(data[tr,:],np.ones((bins[b],)),'valid')
            convo_bins[tr,:] = tmp_cn[0:-1:bins[b]]

        #
        fano[b] = np.mean(np.var(convo_bins,axis=0)/(np.mean(convo_bins,axis=0)+eps))

    #
    return fano

def correlation_binwidths(data, data2, bins, boot_errs=False, nboots=1000, tp1=150, tp2=550, convo_style='same'):
    if (data.shape[0] != data2.shape[0]) or (data.shape[1] != data2.shape[1]):
        raise ValueError('Error in correlation binwidths. Input data dimensions do not agree')

    import numpy as np
    import pdb
    eps = np.finfo(float).eps
    # return fano factor for data, over trial, at bins binwidths
    if boot_errs:
        corrs = np.nan*np.ones((nboots,bins.shape[0]))
        corrs_zentral = np.nan*np.ones((nboots,bins.shape[0]))
        covs = np.nan*np.ones((nboots,bins.shape[0]))
    else:
        corrs = np.nan*np.ones(bins.shape[0])
        corrs_zentral = np.nan*np.ones(bins.shape[0])
        covs = np.nan*np.ones(bins.shape[0])
        
    zscored_response_container = np.nan*np.ones((data.shape[0], 2))
    if boot_errs:
        # bin width loop
        for b in range(bins.shape[0]):
            # trial loop
            for tr in range(data.shape[0]):
                if tr == 0:
                    d_tmp = data[0,tp1:tp2+1]
                    convo_bins  = np.zeros((data.shape[0],d_tmp[0:-1:bins[b]].shape[0]))
                    convo_bins2 = np.zeros((data.shape[0],d_tmp[0:-1:bins[b]].shape[0]))

                #
                tmp_cn = np.convolve(data[tr,:],np.ones((bins[b],)),convo_style)[tp1:tp2]
                tmp_cn2 = np.convolve(data2[tr,:],np.ones((bins[b],)),convo_style)[tp1:tp2]
                convo_bins[tr,:]  = tmp_cn[0:-1:bins[b]]
                convo_bins2[tr,:] = tmp_cn2[0:-1:bins[b]]

            # bootstrap loop
            corrs_all_bins = np.zeros((nboots,convo_bins.shape[1]))
            corrs_all_bins_zentral = np.zeros((nboots,convo_bins.shape[1]))
            covs_all_bins = np.zeros((nboots,convo_bins.shape[1]))
            for dp in range(convo_bins.shape[1]):
                boot_inds = np.random.choice(convo_bins[:,dp].shape[0],(convo_bins[:,dp].shape[0],nboots))
                for bl in range(boot_inds.shape[1]):
                    cz1 = z_score(convo_bins[boot_inds[:,bl],dp])
                    cz2 = z_score(convo_bins2[boot_inds[:,bl],dp])
                    # to use only those trials in which the response was within 3SD of the mean
                    zscored_response_container[:,0] = np.abs(cz1) < 3
                    zscored_response_container[:,1] = np.abs(cz2) < 3
                    ii_dx = np.where(np.sum(zscored_response_container, axis=1) == 2)[0]
                    # avoid divide by zero when unit never spikes
                    if np.std(cz1) == 0 or np.std(cz2) == 0:
                        corrs_all_bins[bl,dp] = 0
                    else:
                        corrs_all_bins[bl,dp] = np.corrcoef(cz1,cz2)[0,1]

                    if np.std(cz1[ii_dx]) == 0 or np.std(cz2[ii_dx]) == 0:
                        corrs_all_bins_zentral[bl,dp] = 0
                    else:
                        corrs_all_bins_zentral[bl,dp] = np.corrcoef(cz1[ii_dx],cz2[ii_dx])[0,1]

                    covs_all_bins[bl,dp] = np.cov(cz1,cz2)[0,1]
                
            corrs[:,b] = np.mean(corrs_all_bins,axis=1)
            corrs_zentral[:,b] = np.mean(corrs_all_bins_zentral,axis=1)
            covs[:,b] = np.mean(covs_all_bins,axis=1)

    else:
        # bin width loop
        for b in range(bins.shape[0]):
            # trial loop
            for tr in range(data.shape[0]):
                if tr == 0:
                    d_tmp = data[0,tp1:tp2+1]
                    convo_bins  = np.zeros((data.shape[0],d_tmp[0:-1:bins[b]].shape[0]))
                    convo_bins2 = np.zeros((data.shape[0],d_tmp[0:-1:bins[b]].shape[0]))

                #
                tmp_cn = np.convolve(data[tr,:],np.ones((bins[b],)),convo_style)[tp1:tp2]
                tmp_cn2 = np.convolve(data2[tr,:],np.ones((bins[b],)),convo_style)[tp1:tp2]
                convo_bins[tr,:]  = tmp_cn[0:-1:bins[b]]
                convo_bins2[tr,:] = tmp_cn2[0:-1:bins[b]]

            corrs_all_bins = np.zeros(convo_bins.shape[1])
            corrs_all_bins_zentral = np.zeros(convo_bins.shape[1])
            covs_all_bins = np.zeros(convo_bins.shape[1])
            for dp in range(convo_bins.shape[1]):
                cz1 = z_score(convo_bins[:,dp])
                cz2 = z_score(convo_bins2[:,dp])
                # to use only those trials in which the response was within 3SD of the mean
                zscored_response_container[:,0] = np.abs(cz1) < 3
                zscored_response_container[:,1] = np.abs(cz2) < 3
                ii_dx = np.where(np.sum(zscored_response_container, axis=1) == 2)[0]
                # avoid divide by zero when unit never spikes
                if np.std(cz1) == 0 or np.std(cz2) == 0:
                    corrs_all_bins[dp] = 0
                else:
                    corrs_all_bins[dp] = np.corrcoef(cz1,cz2)[0,1]

                if np.std(cz1[ii_dx]) == 0 or np.std(cz2[ii_dx]) == 0:
                    corrs_all_bins_zentral[dp] = 0
                else:
                    corrs_all_bins_zentral[dp] = np.corrcoef(cz1[ii_dx],cz2[ii_dx])[0,1]

                covs_all_bins[dp] = np.cov(cz1,cz2)[0,1]
                
            corrs[b] = np.mean(corrs_all_bins)
            corrs_zentral[b] = np.mean(corrs_all_bins_zentral)
            covs[b] = np.mean(covs_all_bins)
        
    #
    return corrs, corrs_zentral, covs

def rasters(clicks, bins, axeli, color = 'black',lw=0.3):
    import numpy as np

    # loop trials
    for tr in range(clicks.shape[0]):
        clicktimes = bins[clicks[tr,:] > 0]
        axeli.vlines(clicktimes, ymin=0+tr,ymax=1+tr,linewidth=lw,colors=color)


def isiCV(raster, first_tp=151, last_tp=551):
    # input is a trials x conditions x time-points array
    CV = np.nan * np.ones((raster.shape[1]))
    tps = np.arange(last_tp - first_tp)

    
    for c in range(raster.shape[1]):
        diffs = []
        for tr in range(raster.shape[0]):
            diffs.append(np.diff(tps[raster[tr,c,first_tp:last_tp] > 0]))
    
        diffs = np.array(diffs)
        if diffs.shape[0] > 2:
            print(diffs)
            CV[c] = np.std(diffs) / np.mean(diffs)

                
    return CV

def ROG(xdata,wc,ws,cg,sg,bs):
    from scipy.special import erf

    Lc = erf(xdata/wc)
    Ls = erf(xdata/ws)

    R = bs + (cg*Lc**2) / (1 + sg*Ls**2)

    return R

def doubleROG(xdata,wc,ws,wcc,wss,cg,ccg,sg,ssg,bs,bs2):
    from scipy.special import erf

    Lc  = erf(xdata/wc)
    Lcc = erf(xdata/wcc)
    Ls  = erf(xdata/ws)
    Lss = erf(xdata/wss)
    
    R = bs + ((cg*Lc**2) / (1 + sg*Ls**2))  + bs2 * ((ccg*Lcc**2)  / (1 + ssg*Lss**2))

    return R

def linfit_variance_to_mean(dada,first_tp,last_tp,count_window=100,nboots=1000,psth_style='same'):
    import numpy as np


