import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
