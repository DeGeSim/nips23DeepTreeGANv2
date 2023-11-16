from torchinfo import summary

from fgsim.monitoring import logger


def log_model(holder):
    for partname, model in holder.models.parts.items():
        try:
            logger.info(f"Model {partname} Summary")
            logger.info(
                summary(
                    model, row_settings=["depth", "var_names"], verbose=0, depth=2
                )
            )
        except Exception:
            pass
