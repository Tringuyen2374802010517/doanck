import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import EmbeddingModel
from dataset import TripletDataset

def compute_topk(model, loader, device, k_list=[1,5]):
    model.eval()
    correct = {k:0 for k in k_list}
    total=0

    with torch.no_grad():
        for a,_,_,label in tqdm(loader):
            a,label = a.to(device),label.to(device)

            _,logits = model(a)
            topk = torch.topk(logits,max(k_list),dim=1).indices

            for i in range(len(label)):
                true = label[i].item()
                preds = topk[i].cpu().numpy()
                for k in k_list:
                    if true in preds[:k]:
                        correct[k]+=1
            total+=len(label)

    print("\nTOP-K RESULT")
    for k in k_list:
        print(f"Top-{k}: {correct[k]/total:.4f}")

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor()
    ])

    ds = TripletDataset("data/val",tf,length=1000)
    loader = DataLoader(ds,batch_size=32)

    model = EmbeddingModel(len(ds.classes)).to(device)
    model.load_state_dict(torch.load("best_model.pth",map_location=device))

    compute_topk(model,loader,device)