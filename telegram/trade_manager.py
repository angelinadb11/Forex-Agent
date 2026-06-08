from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from agents.base import AgentResult
from signal_generator import TradeSignal
from tracking.console import safe_print
from tracking.trade_history import TradeHistoryStore, TradeStatisticsCalculator
from tracking.trade_monitor import ActiveTrade, TradeMonitor

if TYPE_CHECKING:
    from telegram.telegram_bot import TelegramBot

PriceFetcher = Callable[[str], float]
ContextFetcher = Callable[[str, str], dict]

DEFAULT_POLL_INTERVAL = 60.0


class TelegramTradeManager:
    """Sends trade signals to Telegram, stores active trades, and monitors TP/SL."""

    def __init__(
        self,
        price_fetcher: PriceFetcher,
        telegram_bot: TelegramBot | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        history_store: TradeHistoryStore | None = None,
        context_fetcher: ContextFetcher | None = None,
    ) -> None:
        from telegram.telegram_bot import TelegramBot as Bot

        self.telegram_bot = telegram_bot if telegram_bot is not None else Bot.from_env()
        self.price_fetcher = price_fetcher
        self.poll_interval = poll_interval
        self.history_store = history_store or TradeHistoryStore()
        self.stats_calculator = TradeStatisticsCalculator()
        self.monitor = TradeMonitor(
            price_fetcher=price_fetcher,
            history_store=self.history_store,
            stats_calculator=self.stats_calculator,
            telegram_bot=self.telegram_bot,
            context_fetcher=context_fetcher,
        )
        self.active_trades: list[ActiveTrade] = []

    @classmethod
    def from_env(
        cls,
        price_fetcher: PriceFetcher,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        context_fetcher: ContextFetcher | None = None,
    ) -> TelegramTradeManager | None:
        from telegram.telegram_bot import TelegramBot

        bot = TelegramBot.from_env()
        if bot is None:
            return None
        return cls(
            price_fetcher=price_fetcher,
            telegram_bot=bot,
            poll_interval=poll_interval,
            context_fetcher=context_fetcher,
        )

    def publish_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        timeframe: str,
        agent_results: dict[str, AgentResult] | None = None,
        agents_agreement: str = "No",
        news_warning: str | None = None,
    ) -> ActiveTrade:
        """Send the trade signal to Telegram and store it as active."""
        if self.telegram_bot is not None:
            self.telegram_bot.send_trade_signal(
                symbol,
                signal,
                timeframe=timeframe,
                agent_results=agent_results,
                news_warning=news_warning,
            )

        trade = ActiveTrade.from_signal(
            symbol,
            signal,
            agents_agreement=agents_agreement,
            timeframe=timeframe,
            entry_agent_results=agent_results,
        )
        self.active_trades.append(trade)
        self.monitor.active_trades.append(trade)
        return trade

    def monitor_trade(
        self,
        trade: ActiveTrade,
        poll_interval: float | None = None,
        max_polls: int | None = None,
    ) -> ActiveTrade:
        """Poll market price until TP3 or stop loss is hit for one trade."""
        closed_trades = self.monitor_active_trades(
            trades=[trade],
            poll_interval=poll_interval,
            max_polls=max_polls,
        )
        return closed_trades[0] if closed_trades else trade

    def monitor_active_trades(
        self,
        trades: list[ActiveTrade] | None = None,
        poll_interval: float | None = None,
        max_polls: int | None = None,
    ) -> list[ActiveTrade]:
        """Poll all active trades until they close or max polls is reached."""
        targets = trades if trades is not None else self.active_trades
        open_trades = [trade for trade in targets if not trade.closed]
        if not open_trades:
            return []

        interval = self.poll_interval if poll_interval is None else poll_interval
        safe_print()
        safe_print(
            "Telegram Trade Manager monitoring "
            f"{', '.join(trade.symbol for trade in open_trades)} "
            f"(price check every {interval:.0f}s)..."
        )
        closed_trades = self.monitor.monitor_all(
            open_trades,
            poll_interval=interval,
            max_polls=max_polls,
        )
        for trade in closed_trades:
            if trade.closed:
                self.remove_closed_trade(trade)
        return closed_trades

    def publish_and_monitor(
        self,
        symbol: str,
        signal: TradeSignal,
        poll_interval: float | None = None,
        max_polls: int | None = None,
    ) -> ActiveTrade:
        """Send signal to Telegram, store the trade, and start price monitoring."""
        trade = self.publish_signal(symbol, signal)
        closed_trade = self.monitor_trade(trade, poll_interval=poll_interval, max_polls=max_polls)
        return closed_trade

    def remove_closed_trade(self, trade: ActiveTrade) -> None:
        if trade in self.active_trades:
            self.active_trades.remove(trade)
