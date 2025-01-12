import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import csv
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
frocarr = np.zeros((maxeps, len(detp)))
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def convertcsv(bboxfname, bboxpath, detp):
    sliceim, origin, spacing, isflip = load_itk_image(datapath + bboxfname[:-8] + '.mhd')
    origin = np.load(sideinfopath + bboxfname[:-8] + '_origin.npy', mmap_mode='r')
    spacing = np.load(sideinfopath + bboxfname[:-8] + '_spacing.npy', mmap_mode='r')
    resolution = np.array([1, 1, 1])
    extendbox = np.load(sideinfopath + bboxfname[:-8] + '_extendbox.npy', mmap_mode='r')
    pbb = np.load(bboxpath + bboxfname, mmap_mode='r')

    pbbold = np.array(pbb[pbb[:, 0] > detp])
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15
    pbb = nms(pbbold, nmsthresh)

    pbb = np.array(pbb[:, :-1])
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)
    if isflip:
        Mask = np.load(sideinfopath + bboxfname[:-8] + '_mask.npy', mmap_mode='r')
        pbb[:, 2] = Mask.shape[1] - pbb[:, 2]
        pbb[:, 3] = Mask.shape[2] - pbb[:, 3]
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []
    for nk in range(pos.shape[0]):  # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        rowlist.append([bboxfname[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], 1 / (1 + np.exp(-pbb[nk, 0]))])
    return rowlist

def getcsv(detp, eps):
    for ep in eps:
        bboxpath = results_path + str(ep) + '/'
        for detpthresh in detp:
            print('ep', ep, 'detp', detpthresh)
            f = open(bboxpath + 'predanno' + str(detpthresh) + '.csv', 'w')
            print(bboxpath + 'predanno' + str(detpthresh))
            fwriter = csv.writer(f)
            fwriter.writerow(firstline)
            fnamelist = []
            for fname in os.listdir(bboxpath):
                if fname.endswith('_pbb.npy'):
                    fnamelist.append(fname)

            print(len(fnamelist))

            # Remove the multiprocessing part and process the files sequentially
            for fname in fnamelist:
                predanno = convertcsv(fname, bboxpath, detpthresh)
                for row in predanno:
                    fwriter.writerow(row)

            f.close()

if __name__ == '__main__':
    # Just call getcsv directly without using multiprocessing
    getcsv(detp, eps)
