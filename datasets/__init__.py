# datasets/__init__.py
from .aid_dataset import get_aid_dataloaders
from .nwpu_dataset import get_nwpu_dataloaders
from .ucmerced_dataset import get_ucmerced_dataloaders
from .aid_dataset import get_aid_retrieval_loader
from .nwpu_dataset import get_nwpu_retrieval_loader
from .ucmerced_dataset import get_ucmerced_retrieval_loader
__all__ = [
    'get_aid_dataloaders',
    'get_nwpu_dataloaders',
    'get_ucmerced_dataloaders',
    'get_aid_retrieval_loader',
    'get_ucmerced_retrieval_loader',
    'get_nwpu_retrieval_loader'
]