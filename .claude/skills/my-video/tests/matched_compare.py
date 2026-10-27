# TEST-ONLY: same spoken moment in output vs reference (time-mapped through plan segments) + colour stats.
import json, sys, cv2, numpy as np
sys.path.insert(0, '/home/user/video-edit/.claude/skills/my-video/scripts')
from faces import grab_frames
plan_p, out_v, ref_v, png = sys.argv[1:5]
p = json.load(open(plan_p))
def src_t(t):
    for s in p['segments']:
        if s['t0'] <= t < s['t1']: return s['in'] + (t - s['t0'])
outs = [0.25, 2.9, 4.6, 7.6, 10.5, 17.5, 20.6, 27.5, 36.0, 50.5, 60.0, 66.0]
tiles, stats = [], []
def s_(x):
    x = x[:int(480 * 0.55)].reshape(-1, 3).astype(np.float32); y = 0.114*x[:, 0]+0.587*x[:, 1]+0.299*x[:, 2]
    hsv = cv2.cvtColor(x.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return np.array([y.mean(), y.std(), hsv[:, 1].mean(), (x[:, 2]-x[:, 0]).mean()])
for t in outs:
    a = cv2.resize(grab_frames(out_v, [t])[0], (270, 480)); st = src_t(t)
    b = cv2.resize(grab_frames(ref_v, [st])[0], (270, 480))
    if not any(br['t0'] <= t < br['t1'] for br in p['broll']): stats.append((s_(a), s_(b), np.abs(a.astype(float)-b.astype(float))[:int(480*0.55)].mean()))
    cv2.putText(a, 'out %.1fs' % t, (5, 18), 0, 0.5, (0, 255, 255), 1); cv2.putText(b, 'ref %.1fs' % st, (5, 18), 0, 0.5, (0, 255, 255), 1)
    tiles.append(np.vstack([a, b]))
cv2.imwrite(png, np.vstack([np.hstack(tiles[:6]), np.hstack(tiles[6:])]))
o = np.mean([x[0] for x in stats], 0); r = np.mean([x[1] for x in stats], 0)
print('upper-frame stats (luma, contrast, sat, R-B)  output', np.round(o, 1), ' reference', np.round(r, 1), ' pixel MAE %.1f' % np.mean([x[2] for x in stats]))
