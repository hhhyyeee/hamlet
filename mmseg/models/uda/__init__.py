# from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.tent import Tent
# from mmseg.models.uda.dacs_custom import DACS

from run_experiments import CUSTOM
if CUSTOM:
    from mmseg.models.uda.dacs_custom import DACS
    from mmseg.models.uda.dacs_tent import DACS_TENT
    # from mmseg.models.uda.dacs_custom_1 import DACS
else:
    from mmseg.models.uda.dacs import DACS

__all__ = ['DACS', 'Tent', 'DACS_TENT']
# __all__ = ["DACS", "Tent", "CustomDACS"]
