"""Fashion-MNIST transfer learning with a frozen ResNet-18 feature extractor.

The script caches backbone features so CPU execution is practical. If validation
accuracy is below 80%, it performs a documented late-layer fine-tuning pass.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; MODELS=ROOT/'models'; RESULTS=ROOT/'results'; SAMPLES=DATA/'sample_images'
MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True); SAMPLES.mkdir(parents=True,exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
CLASSES=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
BATCH_SIZE=128; LR=1e-3; EPOCHS=8; IMAGE_SIZE=224

transform=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
raw_test_transform=transforms.Compose([transforms.ToTensor()])
train_full=datasets.FashionMNIST(DATA/'FashionMNIST',train=True,download=True,transform=transform)
test=datasets.FashionMNIST(DATA/'FashionMNIST',train=False,download=True,transform=transform)
labels=np.array(train_full.targets)
idx_train,idx_val=train_test_split(np.arange(len(train_full)),test_size=5000,stratify=labels,random_state=42)
train_ds=Subset(train_full,idx_train); val_ds=Subset(train_full,idx_val)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
test_loader=DataLoader(test,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

weights=models.ResNet18_Weights.DEFAULT
backbone=models.resnet18(weights=weights)
backbone.fc=nn.Identity(); backbone.to(DEVICE); backbone.eval()
for p in backbone.parameters(): p.requires_grad=False

def cache(loader,path):
    if path.exists(): return torch.load(path,map_location='cpu')
    xs=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            xs.append(backbone(x.to(DEVICE)).cpu()); ys.append(y)
    out=(torch.cat(xs),torch.cat(ys)); torch.save(out,path); return out
train_feat,train_y=cache(train_loader,DATA/'train_features.pt')
val_feat,val_y=cache(val_loader,DATA/'val_features.pt')
test_feat,test_y=cache(test_loader,DATA/'test_features.pt')

head=nn.Linear(train_feat.shape[1],10).to(DEVICE)
opt=torch.optim.Adam(head.parameters(),lr=LR)
loss_fn=nn.CrossEntropyLoss()
for _ in range(EPOCHS):
    head.train()
    for x,y in DataLoader(TensorDataset(train_feat,train_y),batch_size=512,shuffle=True):
        opt.zero_grad(); out=head(x.to(DEVICE)); loss=loss_fn(out,y.to(DEVICE)); loss.backward(); opt.step()

def eval_head(feat,y):
    head.eval(); preds=[]
    with torch.no_grad():
        for x,_y in DataLoader(TensorDataset(feat,y),batch_size=512): preds.extend(head(x.to(DEVICE)).argmax(1).cpu().numpy())
    return accuracy_score(y.numpy(),preds),np.array(preds)
val_acc,_=eval_head(val_feat,val_y)

# Optional late-layer fine tuning if needed. The script records before/after accuracy.
after_acc=val_acc; finetuned=False
if val_acc < .80:
    # Unfreeze layer4 and classifier, keep early/middle layers frozen.
    backbone=models.resnet18(weights=weights).to(DEVICE)
    backbone.fc=head
    for name,p in backbone.named_parameters(): p.requires_grad=name.startswith('layer4') or name.startswith('fc')
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,backbone.parameters()),lr=1e-4)
    train_loader_ft=DataLoader(train_ds,batch_size=64,shuffle=True,num_workers=0)
    for _ in range(3):
        backbone.train()
        for x,y in train_loader_ft:
            opt.zero_grad(); out=backbone(x.to(DEVICE)); loss=loss_fn(out,y.to(DEVICE)); loss.backward(); opt.step()
    finetuned=True
    head=backbone.fc
    # final evaluation directly on validation/test images
    def direct_eval(loader):
        backbone.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,y in loader: ys.extend(y.numpy()); ps.extend(backbone(x.to(DEVICE)).argmax(1).cpu().numpy())
        return accuracy_score(ys,ps),np.array(ys),np.array(ps)
    after_acc,_,_=direct_eval(val_loader)
    test_acc,y_true,pred=direct_eval(test_loader)
else:
    test_acc,pred=eval_head(test_feat,test_y); y_true=test_y.numpy()

cm=confusion_matrix(y_true,pred,labels=np.arange(10))
report=classification_report(y_true,pred,target_names=CLASSES,output_dict=True,zero_division=0)
# Save a self-contained checkpoint describing the architecture.
torch.save({'arch':'resnet18','classes':CLASSES,'state_dict':(backbone.state_dict() if finetuned else head.state_dict()),'finetuned':finetuned,'image_size':IMAGE_SIZE},MODELS/'product_classifier.pt')
# Export five real test images with obvious labels.
raw_test=datasets.FashionMNIST(DATA/'FashionMNIST',train=False,download=False)
for i in [3,17,42,101,250]:
    img,label=raw_test[i]; Image.fromarray(np.array(img,dtype=np.uint8)).save(SAMPLES/f'{i:05d}_{CLASSES[label].lower().replace(" ","_").replace("/","_")}.png')

out={'device':DEVICE,'split_sizes':{'train':len(train_ds),'validation':len(val_ds),'test':len(test)},'batch_size':BATCH_SIZE,'optimizer':'Adam','learning_rate':LR,'epochs_head':EPOCHS,'feature_extraction_validation_accuracy':float(val_acc),'fine_tuned':finetuned,'final_validation_accuracy':float(after_acc),'test_accuracy':float(test_acc),'confusion_matrix':cm.tolist(),'classification_report':report,'classes':CLASSES}
(RESULTS/'part2_metrics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
