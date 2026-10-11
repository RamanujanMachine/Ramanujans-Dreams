# Light, always-needed names are imported eagerly; the stages (and their heavy
# dependencies) load on first access — see dreamer/_lazy.py.
# ``loading`` stays eager and first: it is light (no heavy deps), its Formatter
# registry relies on import-time registration, and importing it first resolves
# the loading <-> utils.schemes.db_scheme import cycle in the right order.
from . import loading
from dreamer.utils.constants.constant import Constant
from dreamer.utils.constants.ready_made_consts import (
    e, pi, euler_gamma, catalan, gompertz, zeta, sqrt, power, log, pi_squared, calegary
)
from .configs import (
    config,
    sys_config,
    db_config,
    analysis_config,
    search_config,
    logging_config
)
from ._lazy import lazy_exports

__getattr__, __dir__, _lazy_all = lazy_exports(__name__, {
    "analysis": ".analysis",
    "search": ".search",
    "system": ".system",
    "extraction": ".extraction",
    "post_process": ".post_process",
    "System": ".system.system:System",
})
