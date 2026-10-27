# TEST-ONLY: build a raw-like talking-head clip from the reference card:
#  1) inpaint burned-in captions (faces protected), 2) invert the measured grade (so the pipeline's grade isn't applied twice)
import cv2, numpy as np, subprocess, sys, json
src, out, style_p, yunet = sys.argv[1:5]
W, H = 356, 634
st = json.load(open(style_p))
filt = st["color"]["talking_head_filter"]
import re
curves = {c: np.array([[float(a) for a in p.split("/")] for p in re.search(c + r"='([^']+)'", filt).group(1).split()]) for c in "rgb"}
sat = float(re.search(r"saturation=([\d.]+)", filt).group(1))
lut = {c: np.clip(np.interp(np.arange(256) / 255.0, curves[c][:, 1], curves[c][:, 0]) * 255, 0, 255).astype(np.uint8) for c in "rgb"}
det = cv2.FaceDetectorYN.create(yunet, "", (W, H), 0.6)
dec = subprocess.Popen(["ffmpeg","-v","error","-i",src,"-f","rawvideo","-pix_fmt","bgr24","-"], stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r","30","-i","-",
                        "-i",src,"-map","0:v","-map","1:a","-c:v","libx264","-crf","14","-preset","fast","-c:a","aac","-b:a","192k","-shortest",out], stdin=subprocess.PIPE)
k15 = np.ones((15,15),np.uint8)
while True:
    b = dec.stdout.read(W*H*3)
    if len(b) < W*H*3: break
    f = np.frombuffer(b,np.uint8).reshape(H,W,3).copy()
    g = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g,cv2.MORPH_TOPHAT,k15)
    m = (((th>45)&(hsv[:,:,1]<70)) | ((th>35)&(hsv[:,:,0]>=8)&(hsv[:,:,0]<=32)&(hsv[:,:,1]>60)&(hsv[:,:,2]>150))).astype(np.uint8)
    m[:int(H*0.08)] = 0
    _, faces = det.detect(f)
    if faces is not None:
        for x, y, w, h in faces[:, :4]:
            m[max(0,int(y-0.15*h)):int(y+1.1*h), max(0,int(x-0.1*w)):int(x+1.1*w)] = 0
    f = cv2.inpaint(f, cv2.dilate(m, np.ones((5,5),np.uint8)), 4, cv2.INPAINT_TELEA)
    x = f.astype(np.float32); y = (0.114*x[...,0]+0.587*x[...,1]+0.299*x[...,2])[..., None]
    x = np.clip(y + (x - y) / sat, 0, 255).astype(np.uint8)
    f = np.dstack([lut["b"][x[...,0]], lut["g"][x[...,1]], lut["r"][x[...,2]]])
    enc.stdin.write(np.ascontiguousarray(f).tobytes())
enc.stdin.close(); enc.wait()
