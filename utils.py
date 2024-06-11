markershands = ['LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
              'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
               'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
               'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
               'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
              'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
              'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
               'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
               'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
               'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']

markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
          'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
          'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
          'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

facemarks = [str(x) for x in range(478)] #there are 478 points for the face mesh (see google holistic face mesh info for landmarks)

class pt:
    '''The x and y are time series of x and y.'''
    def __init__(self, x, y) -> None:
        self.x = x      # pandas dataframe
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # return [self.x[i], self.y[i]]
        return [self.x.iloc[i], self.y.iloc[i]]

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)

    def __repr__(self) -> str:
        return '({}, {})'.format(self.x, self.y)
    
    def clip(self, clip_bool_list):
        self.x = self.x[clip_bool_list]
        self.y = self.y[clip_bool_list]
        return self

    def downsample(self, rate):
        self.x = self.x.iloc[::rate]
        self.y = self.y.iloc[::rate]
        return self
        

def load_xy_from_df(key: str, df, framesize):
    return pt(df['X_'+key]*framesize[0], df['Y_'+key]*framesize[1])

def load_eye_or_mouth_mesh(df, framesize, indices):
    pts = []
    for i in indices:
        key = str(i)
        pts.append(load_xy_from_df(key, df, framesize))
    return pts

def load_face_mesh(df, framesize):
    # facemarks = [str(x) for x in range(478)] #there are 478 points for the face mesh (see google holistic face mesh info for landmarks)
    pts = []
    for i in facemarks:
        pts.append(load_xy_from_df(i, df, framesize))
    return pts

def load_hand_mesh(df, framesize):
    pts = []
    for i in markershands:
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
    # print(len(trunc_face_mesh))
    return trunc_face_mesh

def convert_time_to_seconds(time_str):
    try:
        # Split the string into minutes and seconds
        minutes, seconds = time_str.split(':')
        
        # Convert minutes and seconds to integers
        minutes = int(minutes)
        seconds = int(seconds)
        
        # Calculate total number of seconds
        total_seconds = minutes * 60 + seconds
        
        return total_seconds
    except ValueError:
        raise ValueError("Invalid time format. Please provide time in 'min:sec' format.")
