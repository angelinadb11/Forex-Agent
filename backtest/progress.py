from __future__ import annotations


class BacktestScanProgress:
    """Print scan progress while iterating historical candles."""

    def __init__(
        self,
        start_index: int,
        end_index: int,
        *,
        update_every: int = 50,
        message_template: str = "Processed {processed}/{total} candles...",
        finish_message: str = "Scan complete.",
    ) -> None:
        self.start_index = start_index
        self.end_index = end_index
        self.update_every = max(1, update_every)
        self.message_template = message_template
        self.finish_message = finish_message
        self.total = max(0, end_index - start_index)
        self._last_reported = 0

    def update(self, index: int) -> None:
        processed = index - self.start_index + 1
        if processed <= 0:
            return
        if processed < self.total and processed - self._last_reported < self.update_every:
            return

        self._last_reported = processed
        print(self._format(processed), flush=True)

    def finish(self) -> None:
        if self.total <= 0:
            return
        if self._last_reported < self.total:
            print(self._format(self.total), flush=True)
        print(self.finish_message, flush=True)

    def _format(self, processed: int) -> str:
        return self.message_template.format(processed=processed, total=self.total)
