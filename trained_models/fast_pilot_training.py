# Three-backbone fruit detection comparison: identical detector, split and settings.
import random, numpy as np, time, json, gc, shutil
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm
import pandas as pd

SEED=42
EPOCHS=5
BATCH_SIZE=8
IMAGE_SIZE=320
assert pathlib.Path('/content/drive/MyDrive').exists(), 'Run the Google Drive connection cell first.'
OUT=pathlib.Path('/content/drive/MyDrive/Fruit_Detection_Fast_Pilot')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'write_check.txt').write_text('Checkpoint folder is writable.')
meta=yaml.safe_load((ROOT/'data.yaml').read_text())
names=meta['names']
if isinstance(names,dict): names=[names[k] for k in sorted(names)]
N=len(names)
assert torch.cuda.is_available(), 'GPU required: connect a T4 runtime.'
device=torch.device('cuda')

def seed_all():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True

class Fruits(Dataset):
 def __init__(self,split,augment=False):
  self.files=sorted(p for p in (ROOT/split/'images').glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp','.bmp'])
  self.augment=augment
  assert self.files, f'No images in {split}'
 def __len__(self): return len(self.files)
 def __getitem__(self,i):
  p=self.files[i]; im=Image.open(p).convert('RGB'); w,h=im.size
  lab=p.parent.parent/'labels'/(p.stem+'.txt')
  boxes=[]; labels=[]
  if lab.exists():
   for line in lab.read_text().splitlines():
    vals=list(map(float,line.split()))
    if not vals: continue
    if len(vals)>5: pts=np.array(vals[1:]).reshape(-1,2); lo=pts.min(0); hi=pts.max(0); vals=[vals[0],*((lo+hi)/2),*(hi-lo)]
    c,x,y,bw,bh=vals
    assert int(c)==c and 0<=c<N, f'Invalid class: {lab}'
    x1=max(0,(x-bw/2)*w); y1=max(0,(y-bh/2)*h)
    x2=min(w,(x+bw/2)*w); y2=min(h,(y+bh/2)*h)
    if x2>x1 and y2>y1: boxes.append([x1,y1,x2,y2]); labels.append(int(c)+1)
  boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
  if self.augment and random.random()<0.5:
   im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
   if len(boxes): boxes[:,[0,2]]=w-boxes[:,[2,0]]
  target={'boxes':boxes,'labels':torch.tensor(labels,dtype=torch.int64),'image_id':torch.tensor(i),'iscrowd':torch.zeros(len(boxes),dtype=torch.int64),'area':(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])}
  return pil_to_tensor(im).float()/255,target

def collate(batch): return tuple(zip(*batch))
train=Fruits('train',True); valid=Fruits('valid'); test=Fruits('test')
keep=sorted(np.random.default_rng(SEED).choice(len(train),size=min(5000,len(train)),replace=False).tolist())
train.files=[train.files[i] for i in keep]
manifest={'experiment':'Fast presentation pilot','seed':SEED,'epochs':EPOCHS,'image_size':IMAGE_SIZE,'batch_size':BATCH_SIZE,'classes':names,'training_images':[p.name for p in train.files],'validation_images':[p.name for p in valid.files],'test_images':[p.name for p in test.files],'note':'Identical training subset for all backbones; original validation and test splits.'}
(OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FAST PILOT:',len(train),'training,',len(valid),'validation,',len(test),'test images; 5 epochs per backbone.',flush=True)

# Check all annotations before a long run.
for ds in [train,valid,test]:
 for i in tqdm(range(len(ds)),desc='Checking labels'): ds[i]

class Backbone(nn.Module):
 def __init__(self,name):
  super().__init__()
  if name=='mobilenetv3':
   m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT); body=m.features; channels=960
  elif name=='efficientnet_b0':
   m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); body=m.features; channels=1280
  else:
   m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT); body=nn.Sequential(*list(m.children())[:-2]); channels=2048
  self.body=body; self.proj=nn.Conv2d(channels,256,1); self.out_channels=256
 def forward(self,x): return self.proj(self.body(x))

def build(name):
 return FasterRCNN(Backbone(name),num_classes=N+1,
  rpn_anchor_generator=AnchorGenerator(((32,64,128,256,512),),((0.5,1.0,2.0),)),
  box_roi_pool=MultiScaleRoIAlign(['0'],7,2),min_size=IMAGE_SIZE,max_size=IMAGE_SIZE,
  box_detections_per_img=100).to(device)

@torch.inference_mode()
def evaluate(model,ds):
 model.eval(); metric=MeanAveragePrecision(box_format='xyxy',class_metrics=True)
 loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,collate_fn=collate)
 for images,targets in tqdm(loader,desc='Evaluating',leave=False):
  preds=model([im.to(device) for im in images])
  metric.update([{k:v.cpu() for k,v in p.items()} for p in preds],[{'boxes':t['boxes'],'labels':t['labels']} for t in targets])
 return metric.compute()

def serial(metrics): return {k:v.tolist() if torch.is_tensor(v) else v for k,v in metrics.items()}

results=[]
for name in ['mobilenetv3','efficientnet_b0','resnet50']:
 seed_all(); model=build(name)
 loader=DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(SEED))
 optimizer=torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.9,weight_decay=0.0005)
 scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.1)
 scaler=torch.amp.GradScaler('cuda')
 start_epoch,best,history=restore_training(name,model,optimizer,scheduler,scaler); started=time.time()
 for epoch in range(start_epoch,EPOCHS):
  model.train()
  # Freeze batch-normalization statistics for small detection batches, equally for all backbones.
  for m in model.modules():
   if isinstance(m,nn.BatchNorm2d): m.eval()
  total=0
  for images,targets in tqdm(loader,desc=f'{name} epoch {epoch+1}/{EPOCHS}'):
   images=[im.to(device) for im in images]
   targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16): loss=sum(model(images,targets).values())
   assert torch.isfinite(loss), f'Nonfinite loss {loss}'
   scaler.scale(loss).backward(); scaler.unscale_(optimizer)
   torch.nn.utils.clip_grad_norm_(model.parameters(),10)
   scaler.step(optimizer); scaler.update(); total+=loss.item()
  scheduler.step(); metrics=evaluate(model,valid); score=float(metrics['map'])
  row={'epoch':epoch+1,'loss':total/len(loader),'val_mAP50_95':score,'val_mAP50':float(metrics['map_50'])}
  history.append(row); print(name,row,flush=True)
  pd.DataFrame(history).to_csv(OUT/f'{name}_history.csv',index=False)
  checkpoint={'model':model.state_dict(),'backbone':name,'classes':names,'epoch':epoch+1,'config':{'image_size':IMAGE_SIZE,'seed':SEED,'epochs':EPOCHS},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict()}
  torch.save(checkpoint,OUT/f'{name}_last.pth')
  if score>best:
   best=score; torch.save(checkpoint,OUT/f'{name}_best.pth')
 training_seconds=time.time()-started
 model.load_state_dict(torch.load(OUT/f'{name}_best.pth',map_location=device,weights_only=False)['model'])
 metrics=evaluate(model,test)
 (OUT/f'{name}_test_metrics.json').write_text(json.dumps(serial(metrics),indent=2))
 # Same first 50 test images; warm up, synchronize, batch size 1; model-only latency.
 sample=[test[i][0].to(device) for i in range(min(50,len(test)))]
 model.eval()
 with torch.inference_mode():
  for _ in range(10): model([sample[0]])
  timings=[]
  for im in sample:
   torch.cuda.synchronize(); t=time.perf_counter(); model([im]); torch.cuda.synchronize(); timings.append(time.perf_counter()-t)
 results.append({'backbone':name,'test_mAP50_95':float(metrics['map']),'test_mAP50':float(metrics['map_50']),'test_mAR100':float(metrics['mar_100']),'latency_ms':np.mean(timings)*1000,'FPS':1/np.mean(timings),'training_seconds':training_seconds,'parameters':sum(p.numel() for p in model.parameters())})
 pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False); display(pd.DataFrame(results))
 del model,optimizer,checkpoint,sample; gc.collect(); torch.cuda.empty_cache()
(OUT/'classes.json').write_text(json.dumps(names))
shutil.make_archive('/content/fruit_fast_pilot','zip',OUT)
print('ALL THREE PILOT MODELS COMPLETE. Download fruit_fast_pilot.zip from the Files panel.')
from google.colab import files
files.download('/content/fruit_fast_pilot.zip')