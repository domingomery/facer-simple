# The idea of this code is to verify is the subject present in the query image 
# is present in the session image.
#
# D. Mery, UC, October, 2018
# http://dmery.ing.puc.cl

from utils import print_definitions, imreadx, face_descriptor, load_fr_model, face_detection, face_scores
from utils import session_descriptors, is_face, show_face

# definitions
img_query   = "Query_6.png"   # query file image (one cropped face)
img_session = "Session_3.png" # session file image (picture of the classroom with students)
fd_method   = 0               # face detection method (0:HOG, 1: CNN)
fr_method   = 1               # face recognition method (0: Dlib, 1: Dlib+, 2: FaceNet)
sc_method   = 0               # 0 cosine similarity, 1 euclidean distance
uninorm     = 1               # 1 means descriptor has norm = 1
theta       = 0.6             # threshold for cosine similarity
print_scr   = 1               # print scores
show_img    = 1               # show images

# init
print("[facer] : ----------------------------------------------------------------") 
print_definitions(fd_method,fr_method,sc_method,uninorm,theta)

# load deep learning model (if any)
print("[facer] : loading face recognition model...") 
fr_model = load_fr_model(fr_method)    

# query image: read, display and description
print("[facer] : reading query image " + img_query + "...") 
face_q    = imreadx(img_query,show_img)
desc_q    = face_descriptor(face_q,fr_method,fr_model,uninorm)

# session image: read, display and face detection
S     = imreadx(img_session,show_img)
print("[facer] : detecting faces in session image " + img_session + "...") 
faces  = face_detection(S,fd_method)
print("[facer] : " + str(len(faces)) + " face(s) found in session image " + img_session)

# computation of descriptors in detected faces of session image
print("[facer] : finding " + img_query + " in session image ...")
D = session_descriptors(S,faces,fr_method,fr_model,uninorm)

# computation of scores between query face and every face of the session
t = face_scores(D,desc_q,sc_method,print_scr)

# find if query face is in session image
i = is_face(t,theta,sc_method)
if i>=0:
    show_face(S,faces[i],i,show_img)
    print("[facer] : face #" + str(i) + " was detected with score = " + str(t[i]))
else:
    print("[facer] : face in query image not detected in session image")
