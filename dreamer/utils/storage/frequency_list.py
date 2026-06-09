from typing import Callable, Any, Optional


class FrequencyList:
    """
    A self-organizing list optimized for linear scanning.
    Items accessed most frequently bubble to the top.

    Optional cross-process sharing
    ------------------------------
    When a *shared_log* is supplied (a ``multiprocessing.Manager().list()``
    proxy), every locally-appended value is also pushed to that shared log, and
    new values discovered by *other* processes are pulled into this local list
    before each scan (:meth:`find`).  Each process therefore keeps its own
    in-memory list — so the (potentially expensive) ``matcher`` scans stay fully
    parallel — while the small discovered values propagate through the shared
    log.  This lets the per-shard search workers reuse each other's identified
    ``(p, q)`` relations instead of re-deriving them (e.g. via LIReC) N times.

    The shared log is append-only; per-process eviction (``max_size``) is local
    and unaffected.  ``shared_log=None`` reproduces the original single-process
    behaviour byte-for-byte.
    """

    def __init__(self, max_size: int = 100, shared_log: Optional[Any] = None):
        """
        :param max_size: Maximum number of locally-retained items (LRU-ish:
            the least-frequently-accessed item is evicted first).
        :param shared_log: Optional ``multiprocessing.Manager().list()`` proxy
            shared across processes.  ``None`` keeps the cache process-local.
        """
        self.items = []  # List of [value, frequency]
        self.max_size = max_size
        self.shared_log = shared_log
        self._synced_len = 0  # How many shared_log entries we've folded in.

    def _add_local(self, value: Any) -> bool:
        """Insert *value* into the local list (dedup + eviction).

        :return: ``True`` if the value was newly inserted, ``False`` if it was
            already present.
        """
        for item in self.items:
            if item[0] == value:
                return False

        if len(self.items) >= self.max_size:
            self.items.pop()  # Remove least frequent (last item)

        # Insert at end with freq=0
        self.items.append([value, 0])
        return True

    def _sync_from_shared(self) -> None:
        """Fold any values appended to *shared_log* by other processes in.

        Reads the shared proxy once (a single IPC round-trip) and advances the
        cursor so each shared entry is folded in at most once.  Entries already
        present locally are skipped by :meth:`_add_local`'s dedup.
        """
        if self.shared_log is None:
            return
        try:
            snapshot = list(self.shared_log)
        except Exception:
            # A torn-down manager (end of shard search) must never break a scan.
            return
        if len(snapshot) <= self._synced_len:
            self._synced_len = len(snapshot)
            return
        for value in snapshot[self._synced_len:]:
            self._add_local(value)
        self._synced_len = len(snapshot)

    def append(self, value: Any):
        """Adds a new item to the cache (starts with freq=0).

        When a *shared_log* is configured, a newly-inserted value is also
        published to it so other processes can reuse it.
        """
        added = self._add_local(value)
        if added and self.shared_log is not None:
            try:
                self.shared_log.append(value)
            except Exception:
                # Manager unavailable (shutdown) — keep the local entry; sharing
                # is a best-effort accelerator, never a correctness requirement.
                pass

    def find(self, matcher: Callable[[Any], bool]) -> Optional[Any]:
        """
        Scans list. If matcher(value) is True, increments freq
        and bubbles the item up. Returns the value.

        Pulls in any cross-process discoveries first when a shared log is set.
        """
        self._sync_from_shared()
        for i, item in enumerate(self.items):
            # 1. Check condition (The expensive part)
            if matcher(item[0]):

                # 2. Increment Frequency
                item[1] += 1

                # 3. Bubble Up (The optimization)
                # Swap with left neighbor if this item has higher frequency
                curr_idx = i
                while curr_idx > 0 and self.items[curr_idx][1] > self.items[curr_idx - 1][1]:
                    # Python swap is atomic and fast
                    self.items[curr_idx], self.items[curr_idx - 1] = \
                        self.items[curr_idx - 1], self.items[curr_idx]
                    curr_idx -= 1
                return item[0]
        return None
