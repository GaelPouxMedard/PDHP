import numpy as np
from copy import deepcopy as copy
import seaborn as sns
import matplotlib.pyplot as plt
from Evaluation import computeResultsMultiprocess, loadFit, loadData
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits import mplot3d
import os
import re
from utils import *
from wordcloud import WordCloud
import multidict as multidict
from PIL import Image
import datetime
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, variance**0.5

## ============= SYNTHETIQUE =============

def readResults():
    arrayRes = []
    with open("results.txt", "r") as f:
        for num, line in enumerate(f):
            if num>1000:
                pass
                #break
            params, res1, res2 = line.replace("\n", "").split("\t")
            nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS, pIter = eval(params)
            res1 = res1.replace("nan", "np.nan")
            K, NMITxt, NMITmp, AdjRandTxt, AdjRandTmp, VmeasTxt, VmeasTmp, LogL = eval(res1)
            Homo_meas_txt, Compl_meas_txt, V_meas_txt = VmeasTxt
            Homo_meas_tmp, Compl_meas_tmp, V_meas_Tmp = VmeasTmp
            res2 = res2.replace("nan", "np.nan")
            try:
                MAE, MJS = eval(res2)
            except:
                MAE, MJS, MJSBL = eval(res2)
            #print(nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS, pIter)
            #print(K, NMITxt, NMITmp, AdjRandTxt, AdjRandTmp, Homo_meas_txt, Compl_meas_txt, V_meas_txt, Homo_meas_tmp, Compl_meas_tmp, V_meas_Tmp, LogL)
            #print(MAE, MJS)

            line = [nbClasses, lg, overlap_voc, overlap_temp, r, perc_rand, vocPerClass, wordsPerEvent, run, runDS, pIter,
                    K, NMITxt, NMITmp, AdjRandTxt, AdjRandTmp, Homo_meas_txt, Compl_meas_txt, V_meas_txt, Homo_meas_tmp, Compl_meas_tmp, V_meas_Tmp, LogL,
                    MAE, MJS]
            arrayRes.append(line)
    arrayRes = np.array(arrayRes)

    return arrayRes

def filterResults(arrayRes, tupsCond, dicKeys):
    arrayFiltered = copy(arrayRes)
    for name, cond in tupsCond:
        ind = arrayFiltered[:, dicKeys[name]]==cond
        arrayFiltered = arrayFiltered[ind]

    return arrayFiltered

def getMeanRunRunDSPart(arrayFiltered, dicKeys):
    aRes = []
    indics = np.unique(arrayFiltered[:, (dicKeys["run"], dicKeys["runDS"], dicKeys["pIter"])], axis=0)
    for run, runDS, pIter in indics:
        if run != 2:
            pass
            #continue
        tupsCondRun = [("run", run), ("runDS", runDS), ("pIter", pIter)]
        a = filterResults(arrayFiltered, tupsCondRun, dicKeys)

        try:
            aRes.append(a[0, dicKeys["K"]:])
        except Exception as e:
            print("Exception 1", e)
            pass
    '''
    for runDS in set(list(arrayFiltered[:, dicKeys["runDS"]])):
        for pIter in set(list(arrayFiltered[:, dicKeys["pIter"]])):
            tupsCondRun = [("runDS", runDS), ("pIter", pIter)]
            a = filterResults(copy(arrayFiltered), tupsCondRun, dicKeys)

            try:
                aRes.append(a[0, dicKeys["K"]:])
            except Exception as e:
                print("Exception 1", e)
                pass
    '''
    try:
        aResMean = np.mean(aRes, axis=0)
        aResStd = np.std(aRes, axis=0)
    except Exception as e:
        print("Exception 2", e)
        return np.zeros((len(dicKeys)-dicKeys["K"])), np.zeros((len(dicKeys)-dicKeys["K"]))
    return aResMean, aResStd

def getDicRes(arrayFiltered, dicKeys, keys):
    dicRes = {}
    dicStd = {}
    i = 0
    indics = np.unique(arrayFiltered[:, (dicKeys["overlap_temp"], dicKeys["overlap_voc"], dicKeys["perc_rand"], dicKeys["r"])], axis=0)
    lg = len(indics)
    for overlap_temp, overlap_voc, perc_rand, r in indics:
        i+=1
        #if i%100==0: print(i*100./lg, "%")
        if overlap_temp not in dicRes: dicRes[overlap_temp]={}
        if overlap_voc not in dicRes[overlap_temp]: dicRes[overlap_temp][overlap_voc]={}
        if perc_rand not in dicRes[overlap_temp][overlap_voc]: dicRes[overlap_temp][overlap_voc][perc_rand]={}
        if r not in dicRes[overlap_temp][overlap_voc][perc_rand]: dicRes[overlap_temp][overlap_voc][perc_rand][r]={}
        if overlap_temp not in dicStd: dicStd[overlap_temp]={}
        if overlap_voc not in dicStd[overlap_temp]: dicStd[overlap_temp][overlap_voc]={}
        if perc_rand not in dicStd[overlap_temp][overlap_voc]: dicStd[overlap_temp][overlap_voc][perc_rand]={}
        if r not in dicStd[overlap_temp][overlap_voc][perc_rand]: dicStd[overlap_temp][overlap_voc][perc_rand][r]={}

        tupsCond = [("overlap_temp", overlap_temp), ("overlap_voc", overlap_voc), ("perc_rand", perc_rand), ("r", r)]
        arrayRefiltered = filterResults(arrayFiltered, tupsCond, dicKeys)
        aResAllRunsMean, aStdAllRunsMean = getMeanRunRunDSPart(arrayRefiltered, dicKeys)
        for key in keys[dicKeys["K"]:]:
            dicRes[overlap_temp][overlap_voc][perc_rand][r][key] = aResAllRunsMean[dicKeys[key]-dicKeys["K"]]
            dicStd[overlap_temp][overlap_voc][perc_rand][r][key] = aStdAllRunsMean[dicKeys[key]-dicKeys["K"]]

    '''
    for overlap_temp in sorted(set(list(arrayFiltered[:, dicKeys["overlap_temp"]])), reverse=True):
        dicRes[overlap_temp] = {}
        for overlap_voc in sorted(set(list(arrayFiltered[:, dicKeys["overlap_voc"]]))):
            dicRes[overlap_temp][overlap_voc] = {}
            for perc_rand in sorted(set(list(arrayFiltered[:, dicKeys["perc_rand"]]))):
                dicRes[overlap_temp][overlap_voc][perc_rand] = {}
                for r in sorted(set(list(arrayFiltered[:, dicKeys["r"]]))):
                    dicRes[overlap_temp][overlap_voc][perc_rand][r] = {}
                    for run in set(list(arrayFiltered[:, dicKeys["run"]])):
                        dicRes[overlap_temp][overlap_voc][perc_rand][r][run] = {}
                        tupsCond = [("overlap_temp", overlap_temp), ("overlap_voc", overlap_voc), ("perc_rand", perc_rand), ("r", r), ("run", run)]
                        arrayRefiltered = filterResults(arrayFiltered, tupsCond, dicKeys)
                        aResAllRunsMean = getMeanRunRunDSPart(arrayRefiltered, dicKeys)
                        #print(perc_rand, overlap_temp, overlap_voc, aResAllRunsMean, tupsCond)
                        for key in keys[dicKeys["K"]:]:
                            dicRes[overlap_temp][overlap_voc][perc_rand][r][run][key] = aResAllRunsMean[dicKeys[key]-dicKeys["K"]]
    '''
    return dicRes, dicStd

def getMetricMatrice(dicRes, metric="NMITxt", tupsCond=[], diffWithDHP=False):
    tupsCond = np.array(tupsCond, dtype=object)
    def testTup(value, key, tupsCond):
        if key in tupsCond[:, 0]:
            if np.abs(value - tupsCond[:, 1][tupsCond[:, 0]==key][0])<1e-5:
                return True
            else:
                return False
        return True

    mat = []
    dims = [list(), list(), list(), list()]
    for overlap_temp in dicRes:
        test = testTup(overlap_temp, "overlap_temp", tupsCond)
        if not test: continue
        dims[0].append(overlap_temp)
        for overlap_voc in dicRes[overlap_temp]:
            test = testTup(overlap_voc, "overlap_voc", tupsCond)
            if not test: continue
            dims[1].append(overlap_voc)
            for perc_rand in dicRes[overlap_temp][overlap_voc]:
                test = testTup(perc_rand, "perc_rand", tupsCond)
                if not test: continue
                dims[2].append(perc_rand)
                for r in dicRes[overlap_temp][overlap_voc][perc_rand]:
                    test = testTup(r, "r", tupsCond)
                    if not test: continue
                    dims[3].append(r)
                    val = dicRes[overlap_temp][overlap_voc][perc_rand][r][metric]
                    if diffWithDHP: val -= dicRes[overlap_temp][overlap_voc][perc_rand][1.][metric]
                    mat.append(val)

    labs = [l for l in dims if len(set(l))>1]
    for d in range(len(labs)):
        for l in range(len(labs[d])):
            labs[d][l]=str(labs[d][l])

    dims = [len(set(l)) for l in dims if len(set(l))>1]
    mat = np.reshape(mat, dims)

    return mat, labs

def plotRes(folder, dicRes, dicStd, XP):
    # tupsCond = [("r", 0.), ("overlap_temp", 0.5), ("overlap_voc", 0.5), ("perc_rand", 0.)]  # Choose n constraint for a (4-n)D tensor
    if XP==1:  # NMI, r vs ovlerap_temp, reste fixé
        for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
            tupsCond = [("overlap_voc", overlap_voc), ("perc_rand", 0.)]
            mat, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=True)
            mat = mat[:, :15]
            labs[1] = labs[1][:15]
            ax = sns.heatmap(mat, cmap="seismic", square=True, center=0., cbar_kws={"shrink": 0.22, "label":"NMI(r)-NMI(1)"}, vmin=-0.35, vmax=0.35)
            ax.set_xticklabels(labs[1])
            ax.set_xlabel("r")
            ax.set_yticklabels(labs[0])
            ax.set_ylabel("Hawkes intensities overlap")
            ax.invert_yaxis()
            ax.set_title(f"Vocabulary overlap: {overlap_voc}")
            plt.tight_layout()
            plt.savefig(folder+f"{XP}_OL_voc={overlap_voc}.pdf")
            plt.close()

    if XP==2:  # NMI, r vs ovlerap_voc, reste fixé
        for overlap_temp in [0., 0.3, 0.5, 0.7]:
            tupsCond = [("overlap_temp", overlap_temp), ("perc_rand", 0.)]
            mat, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=True)
            mat = mat[:, :15]
            labs[1] = labs[1][:15]
            ax = sns.heatmap(mat, cmap="seismic", square=True, center=0., cbar_kws={"shrink": 0.26, "label":"NMI(r)-NMI(1)"}, vmin=-0.35, vmax=0.35)
            ax.set_xticklabels(labs[1])
            ax.set_xlabel("r")
            ax.set_yticklabels(labs[0])
            ax.set_ylabel("Textual overlap")
            ax.invert_yaxis()
            ax.set_title(f"Hawkes intensities overlap: {overlap_temp}")
            plt.tight_layout()
            plt.savefig(folder+f"{XP}_OL_temp={overlap_temp}.pdf")
            plt.close()

    if XP==3:  # NMI_txt, NMI_temp vs perc_rand, r fixé
        for r in [0., 0.4, 0.8, 1., 1.5, 2., 3., 5., 7.5, 10.]+[0.2, 0.6, 0.9, 1.1, 1.3, 2.5, 4., 6., 8.5]:
            tupsCond = [("r", r), ("overlap_temp", 0.), ("overlap_voc", 0.)]
            try:
                matTmp, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=False)
                matTxt, labs = getMetricMatrice(dicRes, "NMITxt", tupsCond, diffWithDHP=False)

                plt.plot([float(l) for l in labs[0]], matTmp, label="NMI temporal clusters")
                plt.plot([float(l) for l in labs[0]], matTxt, label="NMI textual clusters")
                plt.xlabel("% observations with random textual cluster")
                plt.ylabel("NMI")
                plt.title(f"r: {r}")
                plt.ylim([0,1])
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder+f"{XP}_r={np.round(r, 2)}.pdf")
                plt.close()
            except Exception as e:
                print("EXCEPTION", e)
                continue

    if XP==4:  # NMI_txt, NMI_temp vs r, perc_rand fixé
        for perc_rand in np.linspace(0.1, 1, 10):
            tupsCond = [("perc_rand", perc_rand), ("overlap_temp", 0.), ("overlap_voc", 0.)]
            try:
                matTmp, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=False)
                matTxt, labs = getMetricMatrice(dicRes, "NMITxt", tupsCond, diffWithDHP=False)
                plt.plot([float(l) for l in labs[0]], matTmp, label="NMI temporal clusters")
                plt.plot([float(l) for l in labs[0]], matTxt, label="NMI textual clusters")
                plt.xlabel("r")
                plt.ylabel("NMI")
                plt.title(f"% observations with random textual cluster: {perc_rand}")
                plt.ylim([0,1])
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder+f"{XP}_perc_rand={np.round(perc_rand, 2)}.pdf")
                plt.close()
            except Exception as e:
                print("EXCEPTION", e)
                continue

    if XP==5:  # All metrics together
        for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
            for overlap_temp in [0., 0.3, 0.5, 0.7]:
                tupsCond = [("perc_rand", 0.), ("overlap_temp", overlap_temp), ("overlap_voc", overlap_voc)]
                for key in ["NMITxt", "AdjRandTxt", "V_meas_txt"]:
                    try:
                        mat, labs = getMetricMatrice(dicRes, key, tupsCond, diffWithDHP=False)
                    except Exception as e:
                        print("EXCEPTION", e)
                        continue
                    plt.plot([float(l) for l in labs[0]], mat, "-", label=key)
                plt.xlabel("r")
                plt.ylabel("Metrics")
                plt.title(f"Vocabulary overlap: {overlap_voc} - Hawkes intensities overlap: {overlap_temp} - % random cluster assignment: {0}")
                plt.ylim([0,1])
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder+f"{XP}_OL_voc={overlap_voc}_OL_temp={overlap_temp}.pdf")
                plt.close()

    if XP==6:  # All metrics 3D
        tupsCond = [("perc_rand", 0.)]
        for key in ["NMITxt", "AdjRandTxt", "V_meas_txt"]:
            mat, labs = getMetricMatrice(dicRes, key, tupsCond, diffWithDHP=True)

            x = [float(l) for l in labs[0]]
            y = [float(l) for l in labs[1]]
            z = [float(l) for l in labs[2]]

            fig = plt.figure(figsize = (5, 3.5))
            ax = plt.axes(projection ="3d")


            maxMat, minMat = np.max(mat), np.min(mat)
            #mat=(mat-minMat)/(maxMat-minMat)
            cmap = cm.get_cmap('seismic')

            nnz = mat.nonzero()
            for x_ind, y_ind, z_ind, v in zip(*nnz, mat[nnz]):
                if key in ["MJS", "MAE"]:
                    rgba = [cmap(0.5+1-v)]
                    transp = 1-v
                else:
                    rgba = [cmap(0.5+v)]
                    transp = v
                transp = transp**0
                xi, yi, zi = x[x_ind], y[y_ind], z[z_ind]
                ax.scatter3D(xi, yi, zi, s=50, c=rgba, alpha=transp)

            if key in ["MJS", "MAE"]:
                norm = mpl.colors.Normalize(vmin=maxMat,vmax=minMat)
            else:
                norm = mpl.colors.Normalize(vmin=minMat,vmax=maxMat)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label(key, rotation=-90)

            ax.set_xlim([0.7*1.05, 0.])
            ax.set_ylim([0., 0.9*1.05])
            ax.set_zlim([0., 10.*1.05])
            ax.set_xlabel("Hawkes intensities overlap")
            ax.set_ylabel("Vocabulary overlap")
            ax.set_zlabel('r')
            ax.set_title(f"{key}")
            plt.tight_layout()
            plt.savefig(folder+f"{XP}_metric={key}.pdf")
            plt.close()

    if XP==7:  # Divergence metrics
        for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
            for overlap_temp in [0., 0.3, 0.5, 0.7]:
                tupsCond = [("perc_rand", 0.), ("overlap_temp", overlap_temp), ("overlap_voc", overlap_voc)]
                for key in ["MJS", "MAE"]:
                    try:
                        mat, labs = getMetricMatrice(dicRes, key, tupsCond, diffWithDHP=False)
                        if key=="LogL":
                            mat = (mat-np.min(mat))/(np.max(mat)-np.min(mat))
                    except Exception as e:
                        print("EXCEPTION", e)
                        continue
                    plt.plot([float(l) for l in labs[0]], mat, "-", label=key)
                plt.xlabel("r")
                plt.ylabel("Metrics")
                plt.title(f"Vocabulary overlap: {overlap_voc} - Hawkes intensities overlap: {overlap_temp} - % random assignments: {0}")
                #plt.ylim([0,1])
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder+f"{XP}_OL_voc={overlap_voc}_OL_temp={overlap_temp}.pdf")
                plt.close()

    if XP==8:  # All metrics 3D
        tupsCond = [("perc_rand", 0.)]
        for key in ["LogL", "NMITxt", "AdjRandTxt", "V_meas_txt", "MJS", "MAE"]:
            mat, labs = getMetricMatrice(dicRes, key, tupsCond, diffWithDHP=False)

            x = [float(l) for l in set(labs[0])]
            y = [float(l) for l in set(labs[1])]
            z = [float(l) for l in set(labs[2])]

            fig = plt.figure(figsize = (5, 3.5))
            ax = plt.axes(projection ="3d")


            maxMat, minMat = np.max(mat), np.min(mat)
            mat=(mat-minMat)/(maxMat-minMat)
            cmap = cm.get_cmap('Blues')

            nnz = mat.nonzero()
            arrR = []

            x, y = np.array(list(sorted(x))), np.array(list(sorted(y)))
            X, Y = np.meshgrid(x, y)
            for x_ind in range(len(x)):
                for y_ind in range(len(y)):
                    ind = np.where(mat[x_ind, y_ind]==np.max(mat[x_ind, y_ind]))[0]
                    bestR = np.array(z)[ind][0]
                    arrR.append(bestR)
                    ax.scatter3D(x[x_ind], y[y_ind], bestR, s=20, alpha=0.5, c="r")
            Z = np.array(arrR).reshape((len(x), len(y)))

            surf = ax.plot_surface(X.T, Y.T, Z, cmap=cmap,
                                   linewidth=0, antialiased=False)

            #ax.set_xlim([0.7*1.05, 0.])
            #ax.set_ylim([0., 0.9*1.05])
            #ax.set_zlim([0., 10.*1.05])
            ax.set_xlabel("Hawkes intensities overlap")
            ax.set_ylabel("Vocabulary overlap")
            ax.set_zlabel('r')
            ax.set_title(f"{key}")
            plt.tight_layout()
            plt.savefig(folder+f"{XP}_metric={key}.pdf")
            #plt.show()
            plt.close()

    if XP==9:  # Diff NMI txt tmp
        for perc_rand in np.linspace(0, 1, 11):
            tupsCond = [("perc_rand", perc_rand), ("overlap_temp", 0.), ("overlap_voc", 0.)]
            try:
                matTmp, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=False)
                matTxt, labs = getMetricMatrice(dicRes, "NMITxt", tupsCond, diffWithDHP=False)
                matTmpStd, labs = getMetricMatrice(dicStd, "NMITmp", tupsCond, diffWithDHP=False)
                matTxtStd, labs = getMetricMatrice(dicStd, "NMITxt", tupsCond, diffWithDHP=False)
                x = [float(l) for l in labs[0]]
                plt.figure(figsize=(6,4))
                plt.plot(x, matTxt-matTmp, "o-k", label=r"$\Delta$NMI", markersize=5)
                if "_All" not in folder:
                    plt.fill_between(x, matTxt-matTmp, matTxt-matTmp + (matTmpStd**2+matTxtStd**2)**0.5, color="C0", alpha=0.3)
                    plt.fill_between(x, matTxt-matTmp, matTxt-matTmp - (matTmpStd**2+matTxtStd**2)**0.5, color="C0", alpha=0.3)
                #plt.plot([float(l) for l in labs[0]], matTxt, label="NMI textual clusters")
                topLim, bottomLim = np.max(matTxt-matTmp)*1.1, np.min(matTxt-matTmp)*1.1
                plt.fill_between(x, 0, [topLim for _ in range(len(x))], color="C0", alpha=0.3)
                plt.fill_between(x, 0, [bottomLim for _ in range(len(x))], color="C1", alpha=0.3)
                plt.text(x[-4], topLim*0.7, "Better textual clustering")
                plt.text(x[2], bottomLim*0.7, "Better temporal clustering")
                plt.xlabel("r")
                plt.ylabel(r"$\Delta$NMI")
                plt.title(f"Random cluster assignment: {int(perc_rand*100)}% of observations")
                plt.ylim([bottomLim, topLim])
                plt.xlim([x[0],x[-1]])
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder+f"{XP}_perc_rand={np.round(perc_rand, 2)}.pdf")
                plt.close()
            except Exception as e:
                print("EXCEPTION", e)
                continue

    if XP==10:  # NMI, r vs ovlerap_temp, reste fixé
        for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
            tupsCond = [("overlap_voc", overlap_voc), ("perc_rand", 0.)]
            mat, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=False)
            mat = mat[:, :15]
            labs[1] = labs[1][:15]
            ax = sns.heatmap(mat, cmap="afmhot_r", square=True, cbar_kws={"shrink": 0.22, "label":"NMI(r)"}, vmin=0., vmax=1)
            ax.set_xticklabels(labs[1])
            ax.set_xlabel("r")
            ax.set_yticklabels(labs[0])
            ax.set_ylabel("Hawkes intensities overlap")
            ax.invert_yaxis()
            ax.set_title(f"Vocabulary overlap: {overlap_voc}")
            plt.tight_layout()
            plt.savefig(folder+f"{XP}_OL_voc={overlap_voc}.pdf")
            plt.close()

    if XP==11:  # NMI, r vs ovlerap_voc, reste fixé
        for overlap_temp in [0., 0.3, 0.5, 0.7]:
            tupsCond = [("overlap_temp", overlap_temp), ("perc_rand", 0.)]
            mat, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=False)
            mat = mat[:, :15]
            labs[1] = labs[1][:15]
            ax = sns.heatmap(mat, cmap="afmhot_r", square=True, cbar_kws={"shrink": 0.26, "label":"NMI(r)"}, vmin=0., vmax=1)
            ax.set_xticklabels(labs[1])
            ax.set_xlabel("r")
            ax.set_yticklabels(labs[0])
            ax.set_ylabel("Vocabulary overlap")
            ax.invert_yaxis()
            ax.set_title(f"Hawkes intensities overlap: {overlap_temp}")
            plt.tight_layout()
            plt.savefig(folder+f"{XP}_OL_temp={overlap_temp}.pdf")
            plt.close()

    if XP==12:  # Diff NMI txt tmp
        plt.figure(figsize=(12,8))
        for i, perc_rand in enumerate(np.linspace(0.1, 0.9, 9)):
            tupsCond = [("perc_rand", perc_rand), ("overlap_temp", 0.), ("overlap_voc", 0.)]
            try:
                plt.subplot(3,3,i+1)
                matTmp, labs = getMetricMatrice(dicRes, "NMITmp", tupsCond, diffWithDHP=False)
                matTxt, labs = getMetricMatrice(dicRes, "NMITxt", tupsCond, diffWithDHP=False)
                plt.plot([float(l) for l in labs[0]], matTmp, label="NMI temporal clusters")
                plt.plot([float(l) for l in labs[0]], matTxt, label="NMI textual clusters")
                #plt.xlabel("r")
                #plt.ylabel("NMI")
                #plt.title(f"% observations with random textual cluster: {perc_rand}")
                plt.ylim([0,1])
                #plt.legend()
            except Exception as e:
                print("EXCEPTION", e)
                continue

        plt.tight_layout()
        plt.savefig(folder+f"{XP}_perc_rand_all.pdf")
        plt.close()

def plotMetricsSynth():
    #computeResultsMultiprocess(processes=6, loop=True)
    #pause()

    arrayRes = readResults()

    dicKeys = {}
    keys = ["nbClasses", "lg", "overlap_voc", "overlap_temp", "r", "perc_rand", "vocPerClass", "wordsPerEvent", "run", "runDS", "pIter",
            "K", "NMITxt", "NMITmp", "AdjRandTxt", "AdjRandTmp", "Homo_meas_txt", "Compl_meas_txt", "V_meas_txt", "Homo_meas_tmp", "Compl_meas_tmp", "V_meas_Tmp", "LogL",
            "MAE", "MJS"]
    for i, key in enumerate(keys):
        dicKeys[key]=i


    selectedDatasets = [("wordsPerEvent", 20), ("nbClasses", 2), ("lg", 1500), ("vocPerClass", 1000)]
    arrayFiltered = filterResults(arrayRes, selectedDatasets, dicKeys)


    dicRes, dicStd = getDicRes(arrayFiltered, dicKeys, keys)
    folderOut = f"results/Synth/_All_"
    for XP in range(1, 12):
        if XP in [3, 4, 5, 7, 6, 8, 10, 11]: # 4, 5 et 7 ne sont pertinents que si on considère un seul run
            continue
        plotRes(folderOut, dicRes, dicStd, XP=XP)

    for run in range(10):
        print("Run", run)
        selectedDatasetsRun = [("run", run)]
        arrayFilteredRun = filterResults(arrayFiltered, selectedDatasetsRun, dicKeys)
        dicRes, dicStd = getDicRes(arrayFilteredRun, dicKeys, keys)

        folderOut = f"results/Synth/{run}_"
        for XP in reversed(range(1, 13)):
            if XP in [3, 6, 8]:
                continue
            plotRes(folderOut, dicRes, dicStd, XP=XP)

## ============= REDDIT =============

def readParticlesReddit(undeux=1):
    print("Reading data")
    folderFit = "output/Reddit/"
    folderData = "data/Reddit/"
    listfiles = os.listdir(folderFit)
    txtundeux = ""
    if undeux!=1:
        txtundeux="_"+str(undeux)
        listfiles = [file for file in listfiles if "_particle" in file and "Reddit"+txtundeux+"_" in file and ".txt" in file]
    else:
        listfiles = [file for file in listfiles if "_particle" in file and not ("Reddit_2_" in file or "Reddit_3_" in file) and ".txt" in file]

    dicRes = {}
    r = float(re.findall("(?<=Reddit"+txtundeux+"_)(.*)(?=_particles)", listfiles[0])[0])
    news_items, lamb0, means, sigs, alpha = loadData(folderData, listfiles[0], r)

    indToWd = {}
    lgMin, lgMax = 1e20, 0
    with open(folderData+f"Reddit{txtundeux}_words.txt", "r", encoding="utf-8") as f:
        for line in f:
            ind, wd = line.replace("\n", "").split("\t")
            indToWd[int(ind)] = wd
    for file in listfiles:
        r = float(re.findall(f"(?<=Reddit{txtundeux}_)(.*)(?=_particles)", file)[0])
        if r>=2:
            pass
            #continue
        particles = loadFit(folderFit, file, r)
        dicRes[r] = particles
        if len(particles[0].docs2cluster_ID)<lgMin:
            lgMin = len(particles[0].docs2cluster_ID)
        if len(particles[0].docs2cluster_ID)>lgMax:
            lgMax = len(particles[0].docs2cluster_ID)
        #print(r, len(particles[0].docs2cluster_ID))

    print("LONGUEUR RUN MIN", lgMin, "(lg max :", lgMax, ")")
    for r in dicRes:
        for i in range(len(dicRes[r])):
            dicRes[r][i].docs2cluster_ID = dicRes[r][i].docs2cluster_ID[:lgMin]

    data = [news_items, lamb0, means, sigs, indToWd, dicRes]
    print(dicRes.keys())
    return data

def similarityClustersWords(dicRes, observations, indToWd):
    import gensim.downloader as api
    print("Loading Google News corpus")
    model = api.load("glove-twitter-100")  # word2vec-google-news-300
    print("Loading finished")

    print("Computing similarities")
    simij = {}
    lg = len(indToWd)
    abs = 0
    for i in range(len((indToWd))):
        if i%100==0: print(i*100./lg, "%")
        if indToWd[i] not in simij: simij[indToWd[i]] = {}
        for j in range(i, len(indToWd)):
            if indToWd[j] not in simij[indToWd[i]]: simij[indToWd[i]][indToWd[j]] = 0
            try:
                simij[indToWd[i]][indToWd[j]]=model.similarity(indToWd[i], indToWd[j])
            except Exception as e:
                abs += 1
                print(e)
                pass
    print(abs*100./lg, "% mots absents")

    print("Computing similarities clusters")
    arrR, arrSim = [], []
    for r in dicRes:
        arrSimClusPart = []
        for p in dicRes[r]:
            arrSimClus = []
            for i, c in enumerate(p.docs2cluster_ID):
                wds = observations[observations[:, 0] == i][0, 2]
                wds = wds[0]
                simClus, div = 0, 1e-20
                for wd1 in range(len(wds)):
                    for wd2 in range(wd1, len(wds)):
                        simClus += simij[indToWd[wd1]][indToWd[wd2]]
                        div += 1
                simClus /= div
                arrSimClus.append(simClus)
            arrSimClusPart.append(np.mean(arrSimClus))

        arrSimClusPart = np.array(arrSimClusPart)
        arrR.append(r)
        arrSim.append(np.mean(arrSimClusPart))
        print(r, arrSim, arrSimClusPart[0])



def getLikTxt(cluster, theta0=None):
    cls_word_distribution = np.array(cluster.word_distribution, dtype=int)
    cls_word_count = int(cluster.word_count)

    vocabulary_size = len(cls_word_distribution)
    if theta0 is None:
        theta0 = 0.01

    priors_sum = theta0*vocabulary_size  # ATTENTION SI PRIOR[0] SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!
    log_prob = 0

    cnt = np.bincount(cls_word_distribution)
    un = np.arange(len(cnt))

    log_prob += gammaln(priors_sum)
    log_prob += gammaln(cls_word_count+1)
    log_prob += gammaln(un + theta0).dot(cnt)  # ATTENTION SI PRIOR[0] SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!

    log_prob -= gammaln(cls_word_count + priors_sum)
    log_prob -= vocabulary_size*gammaln(theta0)
    log_prob -= gammaln(cls_word_count+1)

    return log_prob

def getLikTmp(timeseq, alphas, reference_time, bandwidth, base_intensity, max_time):
    Lambda_0 = base_intensity * max_time
    #alphas_times_gtheta = np.sum(alphas * g_theta(timeseq, reference_time, bandwidth, max_time), axis = 1) # shape = (sample number,)
    alphas_times_gtheta = alphas.dot(g_theta(timeseq, reference_time, bandwidth, max_time))

    time_intervals = []
    timeseq = np.array(list(sorted(timeseq)))
    horizon = np.max(reference_time)+3*np.max(bandwidth)

    alphas = alphas.reshape(-1, 1, alphas.shape[-1])
    L = -alphas_times_gtheta - Lambda_0
    for i, ti in enumerate(timeseq):
        time_intervals = ti-timeseq[:i]
        time_intervals = time_intervals[time_intervals<horizon]
        L += np.log(triggering_kernel(alphas, reference_time, time_intervals, bandwidth)+1e-100)

    return L


def getLikelihoods(dicRes, observations, lamb0, means, sigs, undeux=1):
    arrR, arrLikTxt, arrLikTmp, arrLikTxtStd, arrLikTmpStd = [], [], [], [], []
    observations = np.array(observations, dtype=object)
    means = np.array(means)
    sigs = np.array(sigs)
    T = np.max(observations[:, 1])
    for r in sorted(dicRes):
        print("r =", r)
        likTmpPart, likTxtPart = [], []
        for p in dicRes[r]:
            likTmp, likTxt = 0., 0.
            for c in p.clusters:
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                timeseq = np.array(observations[inds, 1], dtype=float)
                alphas = np.array([p.clusters[c].alpha])

                likTmp += getLikTmp(timeseq, alphas, means, sigs, lamb0, T)[0]

                try:
                    l = getLikTxt(p.clusters[c])
                except Exception as e:
                    l = p.clusters[c].txtLikelihood
                likTxt += l

            likTmpPart.append(likTmp)
            likTxtPart.append(likTxt)

        arrR.append(r)
        arrLikTmp.append(np.mean(likTmpPart))
        arrLikTxt.append(np.mean(likTxtPart))
        arrLikTmpStd.append(np.std(likTmpPart))
        arrLikTxtStd.append(np.std(likTxtPart))

    arrLikTxt, arrLikTmp, arrLikTxtStd, arrLikTmpStd = np.array(arrLikTxt), np.array(arrLikTmp), np.array(arrLikTxtStd), np.array(arrLikTmpStd)
    #fig, ax = plt.figure(figsize=(5, 3.5))
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(arrR, arrLikTxt, "C0", label="Textual log-likelihood")
    ax.fill_between(arrR, arrLikTxt, arrLikTxt+arrLikTxtStd, color="C0", alpha=0.3)
    ax.fill_between(arrR, arrLikTxt, arrLikTxt-arrLikTxtStd, color="C0", alpha=0.3)
    ax.set_xlabel("r")
    ax.set_ylabel("Textual log-likelihood", color='C0')
    ax2 = ax.twinx()
    ax2.plot(arrR, arrLikTmp, "C1", label="Temporal log-likelihood")
    ax2.fill_between(arrR, arrLikTmp, arrLikTmp+arrLikTmpStd, color="C1", alpha=0.3)
    ax2.fill_between(arrR, arrLikTmp, arrLikTmp-arrLikTmpStd, color="C1", alpha=0.3)
    ax2.set_ylabel("Temporal log-likelihood", rotation=-90, color='C1', labelpad=12)
    #ax.legend()
    #ax2.legend()
    plt.tight_layout()
    plt.savefig(f"results/Reddit/Reddit_{undeux}_Likelihoods.pdf")
    #plt.show()
    plt.close()



def getFrequencyDictForText(words):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in words:
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict

def makeImage(words):
    #alice_mask = np.array(Image.open("alice_mask.png"))
    text = getFrequencyDictForText(words)

    x, y = np.ogrid[:1000, :1000]
    mask = (x - 500) ** 2 + (y - 500) ** 2 > 500 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color="white", max_words=500, mask=mask, colormap="cividis")
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.tight_layout()
    plt.axis("off")
    return text

def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))/(2 * np.pi * np.power(sig, 2.)) ** 0.5

def kernel(dt, means, sigs, alpha):
    k = gaussian(dt[:, None], means[None, :], sigs[None, :]).dot(alpha)
    return k

def compute_kernel_alltimes(timestamps, means, sigs, alpha, res=1000):
    ranget = np.linspace(np.min(timestamps), np.max(timestamps), res)
    tabvals = []
    maxdt = max(means)+3*max(sigs)
    for t in ranget:
        ev = timestamps[timestamps>t-maxdt]
        ev = ev[ev<t]

        eventsprec = ev
        val = kernel(t - eventsprec, means, sigs, alpha).sum()
        tabvals.append(val)
    tabvals = np.array(tabvals)
    return ranget, tabvals

def getLargestClusters(dicRes, observations, indToWd, undeux=1, kwds=None):
    for r in sorted(dicRes):
        print("r =", r)
        for ip, p in enumerate(dicRes[r]):
            lgClus, selectedClusters = [], []
            for c in p.clusters:
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                selectedClusters.append(c)
                lgClus.append(len(inds))
            selectedClusters = [c for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)]
            lgClus = [l for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)]
            selectedClusters = selectedClusters[:30]
            lgClus = lgClus[:30]
            print(list(sorted(lgClus, reverse=True)))

            for ic, c in enumerate(selectedClusters):
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                timeseq = np.array(np.array(observations, dtype=object)[inds, 1], dtype=float)

                #from SMC_sampling import fitTick
                #fitTick(np.array(observations, dtype=object)[inds], means, sigs, p.clusters[c].alpha)


                fig = plt.figure(figsize=(3.5, 10.5))
                words = []
                for o in np.array(observations, dtype=object)[inds, 2]:
                    for wd, cnt in zip(o[0], o[1]):
                        try:
                            if indToWd[wd]!="til":
                                for _ in range(cnt):
                                    words.append(indToWd[wd])
                        except:
                            continue
                plt.subplot(3,1,1)
                makeImage(words)

                save=True
                if kwds is not None:
                    #print(list(txt.keys()))
                    for kw in kwds:
                        un, cnts = np.unique(words, return_counts=True)
                        #print(kw, un)
                        if kw not in un:
                            save=False
                            break
                        if not cnts[un==kw]>np.median(np.unique(cnts, return_counts=True)[0]):
                            pass
                            save=False
                            break
                        print(ic, c, kw, un[un==kw], cnts[un==kw], np.median(np.unique(cnts, return_counts=True)[0]), np.unique(cnts, return_counts=True))
                    if not save:
                        plt.close()
                        continue
                    else:
                        print("============ FOUND")

                plt.subplot(3,1,2)
                dt = np.linspace(0, np.max(means)+3*np.max(sigs), int(np.max(means)-np.min(means))*10)
                plt.plot(dt, triggering_kernel(p.clusters[c].alpha, means, dt, sigs, donotsum=True))
                plt.ylim([1e-5, 1.])
                plt.semilogx()
                plt.semilogy()
                plt.ylabel(r"$\lambda(t)$")
                plt.xlabel("Time (h)")
                plt.subplot(3,1,3)
                res = int(np.max(timeseq)-np.min(timeseq))*10
                x, y = compute_kernel_alltimes(timeseq, means, sigs, np.array(p.clusters[c].alpha), res=res)
                plt.plot(timeseq, timeseq**0-1.1, "ok", markersize=0.3)
                plt.plot(x, y)
                x_ticks = [x_i for ind, x_i in enumerate(x) if ind%(res/5)==0]
                x_labels = [datetime.datetime.fromtimestamp(float(ts)*3600).strftime("%d/%m/%y") for ts in x_ticks]
                plt.xticks(x_ticks, x_labels, rotation=45, ha="right")
                plt.xlabel(r"Date")
                plt.ylabel(r"Intensity")
                #plt.semilogx()


                plt.tight_layout()
                plt.savefig(f"results/Reddit/Reddit_{undeux}_r={r}_part={ip}_clus={ic}.png")
                if kwds is not None:
                    if save: plt.savefig(f"results/Reddit/Part/Reddit_{undeux}_r={r}_part={ip}_clus={ic}.png")

                #plt.show()
                plt.close()
            break  # Une seule particule

def getKernelsLargestClusters(dicRes, undeux=1):
    tabMeanEntropyfft = []
    tabStdEntropyfft = []
    tabR = []
    for r in sorted(dicRes):
        print("r =", r)
        for ip, p in enumerate(dicRes[r]):
            lgClus, selectedClusters = [], []
            for c in p.clusters:
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                selectedClusters.append(c)
                lgClus.append(len(inds))
            selectedClusters = [c for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)][:30]
            lgClus = [l for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)][:30]
            print(list(sorted(lgClus, reverse=True)))

            plotKernel = False
            if plotKernel:
                tabKernels = []
                dt = np.linspace(0, np.max(means)+3*np.max(sigs), int(np.max(means)*100))
                for ic, c in enumerate(selectedClusters):
                    tabKernels.append(triggering_kernel(p.clusters[c].alpha, means, dt, sigs, donotsum=True))

                meanKernels = np.mean(tabKernels, axis=0)
                stdKernels = np.std(tabKernels, axis=0)
                plt.plot(dt, meanKernels)
                plt.fill_between(dt, meanKernels, meanKernels+stdKernels, color="C0", alpha=0.3)
                plt.fill_between(dt, meanKernels, meanKernels-stdKernels, color="C0", alpha=0.3)
                plt.semilogx()
                plt.ylim([0., 0.7])
                plt.xlabel("Time (h)")
                plt.ylabel(r"$\lambda (t)$")
                plt.savefig(f"results/Reddit/Reddit_{undeux}_r={r}_part={ip}_dispKernels.png")
                #plt.show()
                plt.close()

            tabEntropyfft = []
            weights = []
            from scipy.fft import fft
            #selectedClusters = selectedClusters[:2]  # ====================== TO REMOVE
            for ic, c in enumerate(selectedClusters):
                print(ic)
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                timeseq = np.array(np.array(observations, dtype=object)[inds, 1], dtype=float)
                res = 10000
                x, y = compute_kernel_alltimes(timeseq, means, sigs, np.array(p.clusters[c].alpha), res=res)
                coeff = fft(y)
                coeff = np.abs(coeff[:res//2])
                coeff = coeff/np.sum(coeff)
                ent = np.log(coeff).dot(coeff) / (np.log(1./res))
                tabEntropyfft.append(ent)
                weights.append(len(inds))

            break  # Une seule particule

        tabEntropyfft = np.array(tabEntropyfft)
        weights = np.array(weights)
        mean, std = weighted_avg_and_std(tabEntropyfft, weights)
        tabMeanEntropyfft.append(mean)
        tabStdEntropyfft.append(std)
        tabR.append(r)

    tabR, tabMeanEntropyfft, tabStdEntropyfft = np.array(tabR), np.array(tabMeanEntropyfft), np.array(tabStdEntropyfft)
    plt.plot(tabR, tabMeanEntropyfft, color="C0")
    plt.fill_between(tabR, tabMeanEntropyfft-tabStdEntropyfft, tabMeanEntropyfft+tabStdEntropyfft, alpha=0.3, color="C0")
    plt.tight_layout()
    plt.savefig("results/Reddit/Entropy.pdf")
    #plt.show()
    plt.close()

def repartitionClusNews(observations, dicRes, indToWd):
    l = {}
    obsToVar = []
    lg=len(observations)
    with open("data/Reddit/Reddit_metadata.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i==lg:
                break
            l["id"], l["subreddit"], l["selftext"], l["score"], l["num_crossposts"], l["num_comments"], l["permalink"] = line.replace("\n", "").split("\t")

            obsToVar.append([i, l["id"], l["subreddit"], l["score"], l["num_crossposts"], l["num_comments"], l["permalink"]])

    nbClus = 10
    arrRepr = {}
    for r in dicRes:
        if r not in arrRepr: arrRepr[r]={}
        selectedClusters = []
        lgClus = []
        for c in dicRes[r][0].clusters:
            inds = np.where(np.array(dicRes[r][0].docs2cluster_ID, dtype=int)==int(c))[0]
            selectedClusters.append(c)
            lgClus.append(len(inds))
        selectedClusters = [c for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)]
        lgClus = [l for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)]
        selectedClusters = selectedClusters[:nbClus]
        lgClus = lgClus[:nbClus]
        for i, c in enumerate(dicRes[r][0].docs2cluster_ID):
            if c not in selectedClusters: continue
            if c not in arrRepr[r]: arrRepr[r][c]={}
            index, id, subreddit, score, numcrosspost, numcomments, permalink = obsToVar[i]
            if "subreddit" not in arrRepr[r][c]: arrRepr[r][c]["subreddit"] = {}
            if subreddit not in arrRepr[r][c]["subreddit"]: arrRepr[r][c]["subreddit"][subreddit] = 0
            arrRepr[r][c]["subreddit"][subreddit] += 1

            if "score" not in arrRepr[r][c]: arrRepr[r][c]["score"] = []
            arrRepr[r][c]["score"].append(float(score))

    subreddits = False
    if subreddits:
        nbr = len(arrRepr)
        fig, ax = plt.subplots(figsize=(1.*nbClus, 2*nbr))
        labels = [(i, txt) for i, txt in enumerate(['worldnews', 'news', 'nottheonion', 'inthenews', 'offbeat', 'qualitynews', 'neutralnews', 'open_news', 'truenews'])]
        un_sub, cnt_sub = np.unique(np.array(obsToVar, dtype=object)[:, 2], return_counts=True)
        dicFreq = {un_sub[i]: cnt_sub[i]/np.sum(cnt_sub) for i in range(len(un_sub))}
        i = 0
        for r in arrRepr:
            lg = len(arrRepr[r])
            for c in arrRepr[r]:
                plt.subplot(nbr, lg, i+1)
                if (i+1)%nbClus==1:
                    plt.ylabel(f"r={r}")
                dat = np.array([[txt, arrRepr[r][c]["subreddit"][txt]] for txt in arrRepr[r][c]["subreddit"]], dtype=object)
                dat = [dat[dat[:, 0]==l][0] for i, l in labels if l in dat[:, 0]]
                while len(dat) != len(labels):
                    a = [labels[len(dat)-1][1], 0]
                    dat.append(np.array(a))
                dat = np.array(dat, dtype=object)
                data = np.array(dat[:, 1], dtype=int)
                data = data*100./np.sum(data)
                data = np.array([data[i]/dicFreq[l] for i, (index, l) in enumerate(labels)])  # Adjust for chance
                data = data / np.sum(data)
                labs = dat[:, 0]
                wedges, texts = plt.pie(data, radius=1.)

                if i+1 == lg and False:
                    plt.legend(wedges, labs, fontsize="small", title="Subreddits", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                i+=1

        fig.subplots_adjust(wspace=0., hspace=-0.8)
        plt.savefig("results/Reddit/Piecharts_subreddits_Reddit.pdf")
        plt.close()
        #plt.show()

    score = True
    if score:
        nbr = len(arrRepr)
        fig, ax = plt.subplots(figsize=(1.*nbClus, 2*nbr))
        i = 0
        for r in arrRepr:
            lg = len(arrRepr[r])
            for c in arrRepr[r]:
                plt.subplot(nbr, lg, i+1)
                if (i+1)%nbClus==1:
                    plt.ylabel(f"r={r}")
                print(np.max(arrRepr[r][c]["score"]), np.min(arrRepr[r][c]["score"]))
                data = np.array(arrRepr[r][c]["score"])
                logbins = np.logspace(0, 5, 20)
                plt.hist(data, bins=logbins, density=True)
                plt.semilogx()
                plt.semilogy()
                plt.xlim([0, 100000])
                plt.ylim([0, 1])

                i+=1

        #fig.subplots_adjust(wspace=0., hspace=-0.8)
        plt.savefig("results/Reddit/Piecharts_scores_Reddit.pdf")
        #plt.show()
        plt.close()

def getGoodnessTextClus():
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    for undeux in [1, 2, 3]:
        observations, lamb0, means, sigs, indToWd, dicRes = readParticlesReddit(undeux)
        V = len(indToWd)
        tabR, tabMeanEnt, tabStdEnt, tabSemEnt = [], [], [], []
        for r in sorted(dicRes):
            print("r =", r)
            p = dicRes[r][0]
            lgClus, selectedClusters = [], []
            for c in p.clusters:
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                selectedClusters.append(c)
                lgClus.append(len(inds))
            selectedClusters = [c for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)]
            lgClus = [l for l, c in sorted(zip(lgClus, selectedClusters), reverse=True)]

            X, C = [], []
            tabEnt, weights = [], []
            for c in selectedClusters:
                inds = np.where(np.array(p.docs2cluster_ID, dtype=int)==int(c))[0]
                wordsdist = np.zeros((V))
                for o in np.array(observations, dtype=object)[inds, 2]:
                    wordsdist[o[0]] += o[1]

                weights.append(np.sum(wordsdist))
                #weights.append(1)
                wordsdist = wordsdist[wordsdist!=0]
                wordsdist = wordsdist/np.sum(wordsdist)
                ent = np.log(wordsdist).dot(wordsdist) / (np.log(1./V))
                tabEnt.append(ent)


            tabEnt, weights = np.array(tabEnt), np.array(weights)
            avg, std = weighted_avg_and_std(tabEnt, weights=weights)
            tabR.append(r)
            tabMeanEnt.append(avg)
            tabStdEnt.append(std)
            tabSemEnt.append(std/(len(weights)**0.5))

        if undeux==1:
            lab = "News"
        elif undeux==2:
            lab="TIL"
        else:
            lab="AskScience"
        tabR, tabMeanEnt, tabStdEnt, tabSemEnt = np.array(tabR), np.array(tabMeanEnt), np.array(tabStdEnt), np.array(tabSemEnt)
        plt.plot(tabR, tabMeanEnt, color=f"C{undeux}")
        plt.fill_between(tabR, tabMeanEnt-tabSemEnt, tabMeanEnt+tabSemEnt, alpha=0.3, color=f"C{undeux}", label=lab)
    plt.xlabel("r")
    plt.ylabel("Average entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/Reddit/Entropy_txt.pdf")
    #plt.show()
    plt.close()



if __name__ == "__main__":

    #plotMetricsSynth()
    #pause()

    getGoodnessTextClus()
    #pause()

    undeux = 3
    for undeux in [3, 2, 1]:
        observations, lamb0, means, sigs, indToWd, dicRes = readParticlesReddit(undeux)


        print("Repartition clusters")
        repartitionClusNews(observations, dicRes, indToWd)
        print("Kernels")
        getKernelsLargestClusters(dicRes, undeux)
        print("Likelihoods")
        getLikelihoods(dicRes, observations, lamb0, means, sigs, undeux)
        print("Wordclouds")
        getLargestClusters(dicRes, observations, indToWd, undeux)#, kwds=["notre", "dame", "fire"])


    #similarityClustersWords(dicRes, observations, indToWd)


