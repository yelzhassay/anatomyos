import sys, json, struct, numpy as np, trimesh, fast_simplification
S=sys.argv[1]; out=sys.argv[2]
rows=[l.rstrip('\n').split('\t') for l in open(S+'/bones2.txt')]
seen=set(); meshes=[]
for fid,name in rows:
    if fid in seen: continue
    seen.add(fid)
    try: m=trimesh.load(f'{S}/stl/{fid}.stl',force='mesh')
    except Exception as e: print('fail',fid,e); continue
    m.merge_vertices()
    meshes.append((fid,name,m))
areas=np.array([m.area for _,_,m in meshes]); tot=areas.sum()
allv=np.vstack([m.vertices for _,_,m in meshes]); lo=allv.min(0); hi=allv.max(0)
print('bbox',lo,hi,'faces',sum(len(m.faces) for *_,m in meshes))
parts=[]; buf=bytearray(); ext=(hi-lo).max()
for (fid,name,m),a in zip(meshes,areas):
    import re
    mult=2.4 if re.search(r'frontal|parietal|occipital|temporal|sphenoid|ethmoid|maxilla|mandible|zygomatic|nasal|lacrimal|palatine|concha|vomer|hyoid',name) else 1.5 if 'vertebra' in name or name in ('atlas','axis','sacrum') else 1.0
    target=int(np.clip(a/tot*360000*mult,400,16000))
    v,f=m.vertices,m.faces
    if len(f)>target:
        v,f=fast_simplification.simplify(v.astype(np.float32),f.astype(np.int32),target_reduction=1-target/len(f))
        lo_b,hi_b=m.bounds; bad=((v<lo_b-0.5)|(v>hi_b+0.5)).any(1)|~np.isfinite(v).all(1)
        if bad.any():
            print('fix',fid,name,int(bad.sum()))
            f=f[~bad[f].any(1)]
            used=np.unique(f); remap=-np.ones(len(v),int); remap[used]=np.arange(len(used)); v=v[used]; f=remap[f]
    if len(v)>65535: print('too many verts',fid); continue
    q=np.clip(np.round((v-lo)/ext*65535),0,65535).astype(np.uint16)
    off=len(buf); buf+=q.tobytes()
    ioff=len(buf); buf+=f.astype(np.uint16).tobytes()
    if len(buf)%4: buf+=b'\0'*(4-len(buf)%4)
    parts.append(dict(id=fid,en=name,v=int(len(v)),f=int(len(f)),vo=off,io=ioff))
open(out+'/bones.bin','wb').write(buf)
json.dump(dict(lo=lo.tolist(),ext=float(ext),parts=parts),open(out+'/bones_mesh.json','w'))
print(len(parts),'parts',len(buf)/1e6,'MB', sum(p['f'] for p in parts),'faces')
