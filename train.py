import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from tqdm import tqdm
from dataset import VODataset
from model.network import VisionTransformer
from torch.utils.data import random_split
from functools import partial
import gc
from dataset import VODataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
from config import Config
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter


def seed_set(SEED=42):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print(f"Seed set at {SEED}")


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_dataloaders(config):
    data_dir = config.data_dir

    datasets = [d for d in data_dir.iterdir() if d.is_dir()]
    torch_datasets = []

    for dataset_dir in datasets:
        for cam_num in range(2):
            torch_dataset = VODataset(config, base_dir=dataset_dir, cam_num=cam_num)
            torch_datasets.append(torch_dataset)

    full_dataset = ConcatDataset(torch_datasets)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def get_optimizer(config, len_dataloader, model, warm_up_duration=0.1):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    num_training_steps = len_dataloader * config.epochs
    num_warmup_steps = num_training_steps * warm_up_duration

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler

def compute_loss(pred, gt, cfg, loss_fn):

    pred = torch.reshape(pred, (pred.shape[0], cfg.num_frames-1, 6))
    estimated_angles = pred[:, :, :3].flatten()
    estimated_position = pred[:, :, 3:].flatten()

    gt = torch.reshape(gt, (gt.shape[0], cfg.num_frames-1, 6))
    gt_angles = gt[:, :, 3:].flatten()
    gt_position = gt[:, :, :3].flatten()
 
    loss_angles = 100*loss_fn(gt_angles.float(), estimated_angles)
    loss_position = loss_fn(gt_position.float(), estimated_position)
    loss = (loss_angles + loss_position)
    return loss

def training_validation(
    config,
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    load_best=True,
):
    best_loss = float("inf")
    best_model_path = config.checkpoint_dir / f"best.pth"
    last_model_path = config.checkpoint_dir / f"last.pth"

    if best_model_path.is_file():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # config.global_epoch = checkpoint["epoch"] + 1
        config.global_epoch = 0
        checkpoint["epoch"] = 0

    writer = SummaryWriter("runs")
    device = config.device

    for epoch in range(config.global_epoch, config.global_epoch + config.epochs):
        model.train()
        torch.autograd.set_detect_anomaly(True)
        train_loss = 0.0


        train_batch_iterator = tqdm(
            train_dataloader, desc=f"Processing Epoch {epoch + 1} [Train]"
        )

        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

        for i, (image, pose) in enumerate(train_batch_iterator):
            image = image.to(device)
            pose = pose.to(device)

            pred = model(image)
            loss = compute_loss(pred, pose, config, loss_fn)
            train_loss += loss.item()
            train_batch_iterator.set_postfix(batch_loss=loss.item())

            writer.add_scalar(
                "train_batch_loss", loss.item(), i + epoch * len(train_dataloader)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_dataloader)

        writer.add_scalar("train_loss", train_loss, epoch + 1)
        scheduler.step()

        val_batch_iterator = tqdm(
            val_dataloader, desc=f"Processing Epoch {epoch + 1} [Val]"
        )
        val_loss = 0.0

        with torch.inference_mode():
            for i, (image, pose) in enumerate(val_batch_iterator):
                image = image.to(device)
                pose = pose.to(device)


                pred = model(image)
                loss = compute_loss(pred, pose, config, loss_fn)
                val_loss += loss.item()

                writer.add_scalar(
                    "val_batch_loss", loss.item(), i + epoch * len(val_dataloader)
                )

                val_batch_iterator.set_postfix(batch_loss=loss.item())

        val_loss /= len(val_dataloader)
        writer.add_scalar("val_loss", val_loss, epoch + 1)


        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            config.best_loss = best_loss
            config.best_loss_epoch = epoch

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }

            torch.save(checkpoint, best_model_path)

        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }

        torch.save(checkpoint, last_model_path)


    writer.close()
    config.save_config()

def get_model(config):
    model = VisionTransformer(img_size=config.image_size,
                              num_classes=config.num_classes,
                              patch_size=config.patch_size,
                              embed_dim=config.dim,
                              depth=config.depth,
                              num_heads=config.num_heads,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              drop_rate=0.,
                              attn_drop_rate=config.attn_dropout,
                              drop_path_rate=config.ff_dropout,
                              num_frames=config.num_frames)
    
    if config.pretrained is None:
        print("No pretrained model provided. Training from scratch.")
        return model
    state_dict = torch.load(config.pretrained)
    model.load_state_dict(state_dict["model_state_dict"])

    return model 

def main():
    config = Config()    
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_set()
    flush()
    
    train_dataloader, val_dataloader = get_dataloaders(config)
    model = get_model(config).to(config.device)
    
    
    optimizer, lr_scheduler = get_optimizer(
        config=config, len_dataloader=len(train_dataloader), model=model)

    loss_fn = torch.nn.MSELoss()
    training_validation(
        config,
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        lr_scheduler,
    )


if __name__ == "__main__":
    main()

 