import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from noduleCADEvaluationLUNA16compare import noduleCADEvaluation
import os
import csv
from multiprocessing import Pool
import functools
import SimpleITK as sitk

fold = 9
annotations_filename = "H:/Luna16_Data/DeepLung/preprocess/subset9/annotations9.csv"
annotations_excluded_filename = "C:/Users/lized/PycharmProjects/DeepLung/evaluationScript/annotations/annotations_excluded.csv"
seriesuids_filename = "H:/Luna16_Data/DeepLung/preprocess/subset9/seriesids9.csv"
results_path = "C:/Users/lized/PycharmProjects/DeepLung/detector/results/res18/retrft969/val"
sideinfopath = "H:/Luna16_Data/DeepLung/preprocess/subset9/"
datapath = "H:/Luna16_Data/Extracted/luna/subset9/"

maxeps = 150
eps = range(1, maxeps+1, 1)
detp = [-1.5, -1]
isvis = False
nmsthresh = 0.1
nprocess = 38
use_softnms = False
frocarr = np.zeros((maxeps, len(detp)))
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

# Other functions like `load_itk_image`, `iou`, `nms`, `convertcsv`, `getfrocvalue` go here
def getfrocvalue(results_filename):
    return noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,'./')#vis=False)
def getcsv(detp, eps):
    for ep in eps:
        bboxpath = results_path + str(ep) + '/'
        for detpthresh in detp:
            print('ep', ep, 'detp', detpthresh)
            f = open(bboxpath + 'predanno'+ str(detpthresh) + 'd3.csv', 'w')
            print(bboxpath + 'predanno'+ str(detpthresh) + 'd3.csv')
            fwriter = csv.writer(f)
            fwriter.writerow(firstline)
            fnamelist = []
            for fname in os.listdir(bboxpath):
                if fname.endswith('_pbb.npy'):
                    fnamelist.append(fname)
            print(len(fnamelist))
            predannolist = p.map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthresh), fnamelist)
            for predanno in predannolist:
                for row in predanno:
                    fwriter.writerow(row)
            f.close()

def getfroc(detp, eps):
    maxfroc = 0
    maxep = 0
    for ep in eps:
        bboxpath = results_path + str(ep) + '/'
        predannofnamalist = []
        froclist = []
        for detpthresh in detp:
            filename = bboxpath + 'predanno'+ str(detpthresh) + '.csv'
            froclist.append(getfrocvalue(filename))
        # print(predannofnamalist)
        # froclist.append(getfrocvalue(predannofnamalist[0]))
        if maxfroc < max(froclist):
            maxep = ep
            maxfroc = max(froclist)
        # print(froclist)
        for detpthresh in detp:
            # print(int((ep-eps[0])/(eps[1]-eps[0])), int((detpthresh-detp[0])/(detp[1]-detp[0])))
            frocarr[int((ep-eps[0])/(eps[1]-eps[0])), int((detpthresh-detp[0])/(detp[1]-detp[0]))] = \
                froclist[int((detpthresh-detp[0])/(detp[1]-detp[0]))]
            print('ep', ep, 'detp', detpthresh, froclist[int((detpthresh-detp[0])/(detp[1]-detp[0]))])
    print(maxfroc, maxep)
    return frocarr

# Wrap the code that starts multiprocessing within `if __name__ == '__main__':`
if __name__ == '__main__':
    frocarr = getfroc(detp, eps)

    # Save and plot results
    fig = plt.imshow(frocarr.T)
    plt.colorbar()
    plt.xlabel('# Epochs')
    plt.ylabel('# Detection Prob.')
    xtick = detp
    plt.yticks(range(len(xtick)), xtick)
    ytick = eps
    plt.xticks(range(len(ytick)), ytick)
    plt.title('Average FROC')
    plt.savefig(results_path+'frocavg.png')
    print(results_path+'frocavg.png')
    np.save(results_path+'frocavg.npy', frocarr)
    frocarr = np.load(results_path+'frocavg.npy', 'r')
    froc, x, y = 0, 0, 0
    for i in range(frocarr.shape[0]):
        for j in range(frocarr.shape[1]):
            if froc < frocarr[i,j]:
                froc, x, y = frocarr[i,j], i, j
    print(fold, froc, x, y)
