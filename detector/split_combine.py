import torch
import numpy as np
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw
        
        # pad = [ [0, 0],
        #         [margin, nz * side_len - z + margin],
        #         [margin, nh * side_len - h + margin],
        #         [margin, nw * side_len - w + margin]]
        pad = [
            [0, 0],
            [int(margin), int(nz * side_len - z + margin)],
            [int(margin), int(nh * side_len - h + margin)],
            [int(margin), int(nw * side_len - w + margin)]
        ]
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin
                    # Ensure that the slice indices are integers
                    sz, ez = int(sz), int(ez)
                    sh, eh = int(sh), int(eh)
                    sw, ew = int(sw), int(ew)
                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nzhw

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        # if nzhw==None:
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            int(nz * side_len),
           int(nh * side_len),
            int(nw * side_len),
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][int(margin):int(margin + side_len), int(margin):int(margin + side_len), int(margin):int(margin + side_len)]
                    sz = int(sz)
                    ez = int(ez)
                    sh = int(sh)
                    eh = int(eh)
                    sw = int(sw)
                    ew = int(ew)
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output 
