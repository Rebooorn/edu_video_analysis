class pt:
    def __init__(self, x, y) -> None:
        self.x = x      # pandas dataframe
        self.y = y

    def __getitem__(self, i):
        return [self.x[i], self.y[i]]


def load_xy_from_df(key: str, df, framesize):
    return pt(df['X_'+key]*framesize[0], df['Y_'+key]*framesize[1])

def load_eye_or_mouth_mesh(df, framesize, indices):
    pts = []
    for i in indices:
        key = str(i)
        pts.append(load_xy_from_df(key, df, framesize))
    return pts

def load_face_mesh(df, framesize):
    facemarks = [str(x) for x in range(478)] #there are 478 points for the face mesh (see google holistic face mesh info for landmarks)
    pts = []
    for i in facemarks:
        pts.append(load_xy_from_df(i, df, framesize))
    return pts


lm_to_remove = [
    162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    139, 71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135, 138, 215, 177, 137, 227, 34,
    143, 156, 70, 63, 105, 66, 107, 9, 336, 296, 224, 293, 300, 383, 372, 345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 199, 208, 32, 211, 210, 214, 192, 213, 147, 123, 116
]

def get_truncated_face_connections(all_facemesh):
    # all_facemesh = list(mp_holistic.FACEMESH_TESSELATION)
    # print(len(all_facemesh))

    # filter out some keypoints
    trunc_face_mesh = []
    for m in all_facemesh:
        f = False       # false if ltr not in lm_to_remove
        for ltr in lm_to_remove:
            if ltr in m:
                f = True
        if f is False:
            trunc_face_mesh.append(m)
    print(len(trunc_face_mesh))
    return trunc_face_mesh
