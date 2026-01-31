"""
Dataset Downloader Module
=========================
Responsável por baixar um dataset do Kaggle e movê-lo
para a pasta padrão 'data/' na raiz do projeto.
"""

from pathlib import Path
import shutil
import logging
import kagglehub


# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------
class DatasetDownloadError(Exception):
    """Erro ao baixar ou mover o dataset."""
    pass


# ------------------------------------------------------------------
# Path Resolution
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ------------------------------------------------------------------
# Core Function
# ------------------------------------------------------------------
def download_and_move_dataset(dataset_name: str) -> Path:
    """
    Faz o download de um dataset do Kaggle e move para data/<dataset_name>.

    Parameters
    ----------
    dataset_name : str
        Nome do dataset no formato 'usuario/dataset'.

    Returns
    -------
    Path
        Caminho final do dataset no diretório data/.
    """
    try:
        logger.info("Iniciando download do dataset: %s", dataset_name)

        source_path = Path(kagglehub.dataset_download(dataset_name))
        logger.info("Dataset baixado em: %s", source_path)

        dataset_folder_name = dataset_name.split("/")[-1]
        destination_path = DATA_DIR / dataset_folder_name

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if destination_path.exists():
            logger.info("Dataset já existe em: %s", destination_path)
            return destination_path

        shutil.move(str(source_path), str(destination_path))

        logger.info("Dataset movido para: %s", destination_path)
        return destination_path

    except Exception as error:
        logger.error("Erro no processo de download/movimentação", exc_info=True)
        raise DatasetDownloadError(
            f"Falha ao processar o dataset '{dataset_name}'"
        ) from error


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
def main() -> None:
    DATASET_NAME = "olistbr/brazilian-ecommerce"

    dataset_path = download_and_move_dataset(DATASET_NAME)
    print(f"Dataset disponível em: {dataset_path}")


if __name__ == "__main__":
    main()
