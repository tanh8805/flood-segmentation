import csv, cv2, numpy as np, os
O="/home/tuananh/Documents/flood_segmentation/work/v2/depth_cls_raw"
rows=list(csv.DictReader(open(f"{O}/index.csv")))
vid=[r for r in rows if r["src"]=="video"]; oth=sorted([r for r in rows if r["src"]!="video"],key=lambda r:-float(r["conf"]))[:400-len(vid)]
sel=vid+oth
for i,r in enumerate(sel): r["id"]=i
os.makedirs(f"{O}/sheets",exist_ok=True)
for s in range(0,len(sel),20):
    sheet=np.zeros((720,1280,3),np.uint8)
    for j,r in enumerate(sel[s:s+20]):
        im=cv2.imread(f"{O}/crops/{r['crop']}"); f=min(256/im.shape[1],180/im.shape[0]); g=cv2.resize(im,(max(1,int(im.shape[1]*f)),max(1,int(im.shape[0]*f))))
        cell=np.zeros((180,256,3),np.uint8); cell[:g.shape[0],:g.shape[1]]=g; cv2.putText(cell,str(r["id"]),(3,16),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        y,x=divmod(j,5); sheet[y*180:(y+1)*180,x*256:(x+1)*256]=cell
    cv2.imwrite(f"{O}/sheets/s{s//20:02d}.jpg",sheet,[cv2.IMWRITE_JPEG_QUALITY,85])
with open(f"{O}/selected.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,list(sel[0].keys())); w.writeheader(); w.writerows(sel)
print(len(sel),"crops",len(sel)//20,"sheets")
