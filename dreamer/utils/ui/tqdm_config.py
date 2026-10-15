from tqdm import tqdm
from dreamer.utils.logger import Logger


class SmartTQDM(tqdm):
    """Project progress bar: nests (only the top level stays on screen), logs every
    10 %, and mirrors its count into the run monitor's progress tree (a node per
    bar, nested under the current stage / shard)."""

    _depth = 0

    def __init__(self, *args, **kwargs):
        is_top_level = (SmartTQDM._depth == 0)
        SmartTQDM._depth += 1
        if 'leave' not in kwargs:
            kwargs.update({'leave': is_top_level})

        # Preserve the user-facing description for debug progress logs.
        self.progress_desc = kwargs.get('desc')

        # Determine the step size for 10% jumps if 'total' is known
        self.min_log_step = 0
        known_total = None
        if 'total' in kwargs and kwargs['total']:
            known_total = kwargs['total']
        elif len(args) > 0 and hasattr(args[0], '__len__'):
            known_total = len(args[0])
        if known_total:
            self.min_log_step = max(1, known_total // 10)

        super().__init__(*args, **kwargs)
        from dreamer.utils import monitoring
        name = (self.progress_desc or "progress").strip().rstrip(":").strip()
        self._monitor_node = monitoring.open_node(name, getattr(self, "total", None) or known_total)
        self.last_print_func = Logger.print_func
        Logger.print_func = self.write
        self._last_logged_n = 0

    def update(self, n=1):
        """
        Overrides update to force a flush every 10%.
        """
        displayed = super().update(n)
        self._monitor_node.set(done=self.n)

        # Check if we crossed the next 10% threshold
        if self.min_log_step > 0:
            if (self.n - self._last_logged_n) >= self.min_log_step:
                percent = int(100 * self.n / self.total)
                if self.progress_desc:
                    message = f"{self.progress_desc} - progress: {self.n} / {self.total} ({percent}%)"
                else:
                    message = f"SYSTEM PROGRESS - progress: {self.n} / {self.total} ({percent}%)"
                Logger(message, Logger.Levels.debug).log()
                self._last_logged_n = self.n
        return displayed

    def __iter__(self):
        # tqdm's iterator advances ``n`` without calling ``update`` (and writes it
        # back only when it redraws): count here, after each loop body ran.
        node = self._monitor_node
        done = self.n
        for obj in super().__iter__():
            yield obj
            done += 1
            node.set(done=done)

    def close(self):
        node = getattr(self, "_monitor_node", None)
        if node is not None:
            node.close()
        SmartTQDM._depth -= 1
        Logger.print_func = self.last_print_func
        super().close()
