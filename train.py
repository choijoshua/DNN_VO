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

def axis_angle_to_matrix(rotvec):
    # Clamp the norm to avoid exploding values
    max_norm = 1.0
    norm = torch.norm(rotvec, dim=1, keepdim=True)
    norm = torch.clamp(norm, max=max_norm)
    rotvec = rotvec / (torch.norm(rotvec, dim=1, keepdim=True).clamp(min=1e-8)) * norm

    theta = norm.clamp(min=1e-8)
    axis = rotvec / theta
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        zero, -z,    y,
        z,    zero, -x,
        -y,   x,    zero
    ], dim=1).view(-1, 3, 3)
    I = torch.eye(3, device=rotvec.device).unsqueeze(0)
    theta = theta.view(-1, 1, 1)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R

def matrix_to_axis_angle(R):
    """
    Converts a batch of rotation matrices (B, 3, 3) to axis-angle vectors (B, 3).
    This implementation is numerically stable for small rotation angles by using
    a torch.where to avoid division by very small numbers.
    """
    # Compute the rotation angle
    cos_theta = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (B,)

    # Compute the vector part from the skew-symmetric part of R
    r = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)  # (B, 3)

    sin_theta = torch.sin(theta).unsqueeze(1)  # (B, 1)
    # Create a mask for small angles where theta is nearly zero.
    small_angle = (theta < 1e-3).unsqueeze(1)  # (B, 1)
    # For small angles, use r/2; otherwise, use r/(2*sin(theta))
    axis = torch.where(small_angle, r / 2, r / (2 * sin_theta.clamp(min=1e-6)))
    rotvec = axis * theta.unsqueeze(1)
    return rotvec

def pose_vec_to_mat_torch(pose_vec):
    """
    Converts a batch of 6D pose vectors (B, 6) into 4x4 transformation matrices (B, 4, 4).
    The first 3 elements are translation, the last 3 are an axis-angle rotation.
    """
    trans = pose_vec[:, :3]
    rotvec = pose_vec[:, 3:]
    rot_mat = axis_angle_to_matrix(rotvec)
    T = torch.eye(4, device=pose_vec.device).unsqueeze(0).repeat(pose_vec.shape[0], 1, 1)
    T[:, :3, :3] = rot_mat
    T[:, :3, 3] = trans
    return T

def mat_to_pose_vec_torch(T):
    """
    Converts a batch of 4x4 transformation matrices (B, 4, 4) back to 6D pose vectors (B, 6).
    """
    trans = T[:, :3, 3]
    rotvec = matrix_to_axis_angle(T[:, :3, :3])
    return torch.cat([trans, rotvec], dim=1)

def compute_loss(pred, gt, cfg, loss_fn):
    """
    For each batch, composes the two relative poses to obtain the final transformation
    (from frame 0 to frame 2) for both the predictions and ground truth, and then computes
    the MSE loss between them.
    
    Assumes:
      - pred and gt are of shape (B, (num_frames - 1) * 6)
      - cfg.num_frames == 3 (so two relative poses per sample)
    """
    B = pred.shape[0]
    pred = pred.view(B, cfg.num_frames - 1, 6)
    gt = gt.view(B, cfg.num_frames - 1, 6)

    # Compose predicted relative poses: T_01 * T_12 = T_02
    T01_pred = pose_vec_to_mat_torch(pred[:, 0])
    T12_pred = pose_vec_to_mat_torch(pred[:, 1])
    T02_pred = torch.bmm(T01_pred, T12_pred)
    pose_pred_composed = mat_to_pose_vec_torch(T02_pred)

    # Compose ground truth relative poses
    T01_gt = pose_vec_to_mat_torch(gt[:, 0])
    T12_gt = pose_vec_to_mat_torch(gt[:, 1])
    T02_gt = torch.bmm(T01_gt, T12_gt)
    pose_gt_composed = mat_to_pose_vec_torch(T02_gt)

    loss = loss_fn(pose_gt_composed, pose_pred_composed)
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
            print(f"\n{loss.item()}")
            train_batch_iterator.set_postfix(batch_loss=loss.item())

            writer.add_scalar(
                "train_batch_loss", loss.item(), i + epoch * len(train_dataloader)
            )

            loss.backward()
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

 