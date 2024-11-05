import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sklearn.metrics
from sklearn.model_selection import KFold
from typing import Dict, List, Optional
from transformers import get_cosine_schedule_with_warmup
import albumentations as A
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from ignite.metrics.regression import *
from ignite.engine import *
import ttach as tta



class Cfg:
    model_name             = "maxvit_tiny_tf_512.in1k" # tf_efficientnet_b0, convnext_atto, maxvit_tiny_tf_512.in1k	
    img_size               = 512  
    num_classes            = 1
    in_channels            = 1
    num_epochs             = 20
    lr                     = 1e-4 #
    batch_size             = 8  #
    drop_rate              = 0.1
    drop_path_rate         = 0.0
    iters_to_accumulate    = 1
    grad_clip              = None # None
    optimizer              = "AdamW"
    loss_fn                = "L1Loss"  # L1Loss, MSELoss
    scheduler              = "cosine_schedule_with_warmup" # cosine_schedule_with_warmup, steplr
    adamW_eps              = 1e-08 # 1e-4, 1e-08
    warmup_steps_ratio     = 0.1
    n_data                 = -1 #-1
    n_splits               = 1

    device                 = "cuda"
    use_amp                = False  
    compile                = False
    save_model             = True
    save_epochs            = [20]    
    num_workers            = 4
    exp_name               = "1e4_L1_tta_bs8_flipver_test" 
    seed                   = 0


class NetDataset(Dataset):
    def __init__(self, df, cfg, transform):
        super().__init__()
        self.cfg = cfg
        self.df = df 
        self.transform = transform

        self.train_transforms = A.Compose(
            [   
                # A.Resize(cfg.img_size, cfg.img_size),
                # A.RandomRotate90(p=1), 
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=30, scale_limit=0.2, p=0.75),  
                # A.RandomResizedCrop(
                #     self.cfg.img_size,
                #     self.cfg.img_size,
                #     scale=(0.75, 1.0),
                #     ratio=(0.75, 1.3333333333333),
                #     p=0.5,
                # ),
                ToTensorV2(), # HWC to CHW
            ]
        )
        
        self.val_transforms = A.Compose(
            [   
                # A.Resize(cfg.img_size, cfg.img_size),
                ToTensorV2(),
            ]
        )


    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with rasterio.open("train_images/" + row.filename) as img:
            x = img.read().astype(np.float32)  #[1:4] [[1,2,3,4]]

        x = np.concatenate((x, x[1:4]))
        img_rows = []
        for min_id in range(0, 16, 4):
            img_row = np.hstack([i for i in x[min_id:min_id+4]])
            img_rows.append(img_row) #.transpose(2, 1, 0)
        x = np.vstack(img_rows)

        x = np.where(x < 0, 0, x)
        max, min = 2.8003, 0.0
        x = (x - min) / (max - min + 1e-8)

        # x = x.transpose(1,2,0) #!
        if self.transform == "train":
            transformed = self.train_transforms(image=x)
        else: transformed = self.val_transforms(image=x)
        x = transformed["image"]

        y = torch.tensor(row.target)

        return x, y.float()


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=True, num_classes=self.cfg.num_classes, 
                                       in_chans=self.cfg.in_channels, drop_rate=self.cfg.drop_rate, 
                                       drop_path_rate=self.cfg.drop_path_rate)
        
    def forward(self, x):
        x = self.model(x)
        return x
    

def Datasetloader(df, cfg, fold):
    validloader = None
    train_df = df
    if cfg.n_data != -1:
        df = df.iloc[: cfg.n_data]  
        # df = df.sample(frac=0.5).reset_index(drop=True)
    if cfg.n_splits != 1:
        skf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        train_idx, val_idx = list(skf.split(df, df.target))[fold]
        train_df = df.iloc[train_idx].copy() 
        val_df = df.iloc[val_idx].copy()
        valid_dataset = NetDataset(val_df, cfg, "val") 
        validloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, 
                                 num_workers=cfg.num_workers, pin_memory=True)  
        
    train_dataset = NetDataset(train_df, cfg, "train")
    trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    return trainloader, validloader


def train_one_epoch(model, trainloader, cfg, scaler, optimizer, scheduler, epoch, fold, loss_fn, loss_fn2):
    epoch_loss = 0.0
    model.train()
    for i, (x,y) in enumerate(tqdm(trainloader)): #tqdm
        x, y = x.to(cfg.device), y.to(cfg.device)
        with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
            logits = model(x)
            loss = loss_fn(logits.squeeze(), y) 
            # logits = torch.ones((cfg.batch_size, cfg.num_classes), requires_grad=True).to(cfg.device)
            # if loss_fn2 != None:
            #     loss2 = loss_fn2(logits, y)
            #     loss = (loss + loss2) / 2
        
        loss = loss / cfg.iters_to_accumulate
        scaler.scale(loss).backward() # loss.backward()
        if (i + 1) % cfg.iters_to_accumulate == 0:
            if cfg.grad_clip != None and cfg.use_amp == True:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer) # optimizer.step()
            scaler.update()
            scheduler.step() 
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * x.size(0)

    if (epoch+1) in cfg.save_epochs and cfg.save_model:
        torch.save(model.state_dict(), f"models/{cfg.model_name}_{cfg.exp_name}_fold{fold}_epoch{epoch+1}.pt") 
    
    return epoch_loss


def eval_model(model, validloader, cfg, loss_fn, loss_fn2, epoch):
    epoch_loss = 0.0
    preds, labels = [], []

    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            # tta.Scale(scales=[1, 2, 4]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),        
        ]
    )
    model = tta.ClassificationTTAWrapper(model, transforms)

    model.eval()
    for (x, y) in tqdm(validloader): #tqdm
        x, y = x.to(cfg.device), y.to(cfg.device)
        with torch.inference_mode():  
            with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
                logits = model(x).squeeze()
                loss = loss_fn(logits, y) 
                # if loss_fn2 != None:
                #     loss2 = loss_fn2(logits, y)
                #     loss = (loss + loss2) / 2

        preds.append(logits.detach().cpu())
        labels.append(y.detach().cpu())
        epoch_loss += loss.item() * x.size(0)

    preds, labels = torch.cat(preds), torch.cat(labels)

    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)

    metric = PearsonCorrelation()
    metric.attach(default_evaluator, 'corr')
    state = default_evaluator.run([[preds, labels]])
    score = state.metrics['corr']

    return epoch_loss, score


def train(df, cfg):
    folds_average = 0.0

    for fold in range(cfg.n_splits):
        trainloader, validloader = Datasetloader(df, cfg, fold)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp) # scaler = None
        model = Net(cfg).to(cfg.device) 
        if cfg.compile == True:
            model = torch.compile(model)

        print("-" * 40)
        print(f"Fold {fold+1}/{cfg.n_splits}")

        loss_fn2 = None
        if cfg.loss_fn == "L1Loss":
            loss_fn = torch.nn.L1Loss().to(cfg.device)
        elif cfg.loss_fn == "MSELoss":
            loss_fn = torch.nn.MSELoss().to(cfg.device)
 
        if cfg.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.adamW_eps)
        elif cfg.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        if cfg.scheduler == "cosine_schedule_with_warmup":
            total_steps = len(trainloader) * cfg.num_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=round(total_steps*cfg.warmup_steps_ratio), 
                num_training_steps=total_steps
            )
        elif cfg.scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
            
        for epoch in range(cfg.num_epochs):
            print("-" * 10)
            print(f"Epoch {epoch+1}/{cfg.num_epochs}")

            # Training
            running_loss = train_one_epoch(model, trainloader, cfg, scaler, optimizer, scheduler, epoch, fold, loss_fn, loss_fn2)  
            train_epoch_loss = running_loss / len(trainloader.dataset)
            # print(f"Train Loss: {train_epoch_loss:.4f} ") 

            # Validation
            if validloader != None:
                running_loss, score = eval_model(model, validloader, cfg, loss_fn, loss_fn2, epoch)
                valid_epoch_loss = running_loss / len(validloader.dataset)
                print(f"Train Loss: {train_epoch_loss:.4f} | Valid Loss: {valid_epoch_loss:.4f} | Score: {score:.4f}") 
                if epoch == cfg.num_epochs-1:
                    folds_average += score
                    print(f"Folds average: {folds_average/(fold+1):.4f}")


def main():
    cfg = Cfg()

    seed = cfg.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    if cfg.use_amp == False:
        torch.set_float32_matmul_precision('high')

    df = pd.read_csv("train_answer.csv")

    train(df, cfg)


if __name__ == "__main__":
    main()

