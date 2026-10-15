import sys, os, json, zipfile, collections
sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import HTTPRangeFile
B="https://data.researchdatafinder.qut.edu.au/dataset/9d6f0314-8709-4fa4-838f-c4e5d226713f/resource/"
Z={ # name: (resource id, filename)
 "flooded/Cambogan_20250811_113017":("6e78857b-139d-486b-9c0b-d4496dff554a","cambogan_20250811_113017.zip"),
 "flooded/DairyCreek_20250811_103318":("253eaea7-d9a0-4f18-a7c4-d988ef839db2","dairycreek_20250811_103318.zip"),
 "flooded/Holmview_20250820_130327":("8d70ec22-31bc-4175-a38b-c711ee765db5","holmview_20250820_130327.zip"),
 "flooded/Pullenvale_20250916_124105":("d8b2f04f-22a3-4662-a7fc-c93333cd3d01","pullenvale_20250916_124105.zip"),
 "dry/Cambogan_20250812_122101":("8d541970-c6ec-475f-90f3-d0b159299ecc","cambogan_20250812_122101.zip"),
 "dry/Cambogan_20250812_122339":("141906a0-1d5c-4b7e-bdf0-0ada7c29d9ab","cambogan_20250812_122339.zip"),
 "dry/Cambogan_20250812_122621":("d531cb9a-1be9-4253-ab03-7c144cad6469","cambogan_20250812_122621.zip"),
 "dry/Dairy-Creek_20250812_122954":("d751495d-022d-4930-8d7c-4c4f81ee8fdb","dairy-creek_20250812_122954.zip"),
 "dry/Dairy-Creek_20250812_123312":("3bf369b1-0a4e-47cc-b970-14bd2de8cf0e","dairy-creek_20250812_123312.zip"),
 "dry/Holmview_20250812_120100":("acedc2c6-a65f-4a27-b8a3-6ed7a97079b7","holmview_20250812_120100.zip"),
 "dry/Holmview_20250812_120856":("96e5fb60-4839-4622-a645-7204a0e43e0d","holmview_20250812_120856.zip"),
 "dry/Pullenvale_20250812_134316":("d11e2c04-b1ed-442c-ba8f-1bc27a451b0b","pullenvale_20250812_134316.zip"),
 "dry/Pullenvale_20250812_134524":("0509a45f-f80c-41e6-9620-7a76286b4244","pullenvale_20250812_134524.zip"),
}
def url(k): r,f=Z[k]; return f"{B}{r}/download/{f}"
if __name__=="__main__":
    out={}
    for k in Z:
        zf=zipfile.ZipFile(HTTPRangeFile(url(k)))
        names=zf.namelist()
        fi=sorted(n for n in names if 'front-imgs/' in n and n.endswith('.png'))
        fl=sorted(n for n in names if 'front-labels/' in n and n.endswith('.png'))
        out[k]=dict(front_imgs=fi, front_labels=fl, size=sum(i.file_size for i in zf.infolist() if 'front-' in i.filename))
        print(k, 'files',len(names),'front-imgs',len(fi),'front-labels',len(fl), 'sample', fi[:1], flush=True)
    _d = os.path.join(os.environ.get('FLOOD_WORK', os.path.abspath('work')), 'external', 'fred'); os.makedirs(_d, exist_ok=True)
    json.dump(out, open(os.path.join(_d, 'index.json'), 'w'))
