from pathlib import Path
import pickle


class DeepVOConfig:
    def __init__(self, config_file=None):
        self.device = None
        
        # Model settings
        self.dim = 384
        self.num_frames = 2
        self.image_size = (190, 640)
        self.num_classes = 6 * (self.num_frames - 1)
        self.depth = 12
        self.num_heads = 6
        self.dim_head = 64
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1
        self.num_workers = 4
        self.hidden_size=1000
        
        # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
        # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
        # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
        
        #data
        self.data_dir = Path("/home/undergrad3203/Downloads/data")


        # Training
        self.lr = 1e-5
        self.batch_size = 4
        self.pretrained = None
        self.epochs = 200

        # Checkpoint
        self.checkpoint_dir = Path("checkpoints/DEEPVO")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_loss = float("inf")
        self.best_loss_epoch = 0
        self.global_epoch = 0
        self.config_path = Path("config.pkl")


        if config_file:
            self.config_path = Path(config_file)
            if self.config_path.exists():
                self.load_config()

    def save_config(self):
        with self.config_path as file:
            pickle.dump(self, file)

    def load_config(self):
        if self.config_path.exists():
            with self.config_path.open("rb") as file:
                loaded_config = pickle.load(file)
                self.__dict__.update(loaded_config.__dict__)
        else:
            print(f"Config file {self.config_path} not found.")
        return self