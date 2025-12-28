"""Console dashboard for LLQuant using Rich.

The dashboard polls account info, prices, and positions and renders them in
place using `rich.live.Live`. Credentials are read from the environment so the
file stays secret-free:

	WEEX_API_KEY
	WEEX_SECRET_KEY
	WEEX_PASSPHRASE

Symbols default to the allowed list from `api`, and you can override with
`GUI_SYMBOLS` (comma-separated). Refresh cadence can be tweaked with
`GUI_REFRESH_SECONDS`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
load_dotenv()

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from api import ALLOWED_SYMBOLS
from api.account import get_account_assets
from api.market import get_ticker_price
from api.trade import get_positions


# Defaults can be overridden via env vars to avoid editing code for quick tweaks.
DEFAULT_REFRESH_SECONDS = float(os.getenv("GUI_REFRESH_SECONDS", "5"))


@dataclass
class DashboardState:
	account: Optional[Dict[str, Any]] = None
	prices: Dict[str, Optional[float]] = field(default_factory=dict)
	positions: Dict[str, Any] = field(default_factory=dict)
	last_update: datetime = field(default_factory=datetime.utcnow)
	error: Optional[str] = None


def load_symbols() -> List[str]:
	raw = os.getenv("GUI_SYMBOLS")
	if raw:
		return [s.strip() for s in raw.split(",") if s.strip()]
	return list(ALLOWED_SYMBOLS)


def load_credentials() -> Optional[Dict[str, str]]:
	# Support both naming schemes for convenience.
	api_key = os.getenv("WEEX_API_KEY") or os.getenv("API_Key")
	secret_key = os.getenv("WEEX_SECRET_KEY") or os.getenv("secret_key")
	passphrase = os.getenv("WEEX_PASSPHRASE") or os.getenv("passphrase")
	locale = os.getenv("WEEX_LOCALE") or "en-US"
	if not (api_key and secret_key and passphrase):
		return None
	return {
		"api_key": api_key,
		"secret_key": secret_key,
		"passphrase": passphrase,
		"locale": locale,
	}


def safe_float(value: Any) -> str:
	try:
		return f"{float(value):,.2f}"
	except Exception:
		return str(value)


def normalize_positions(raw: Any) -> List[Dict[str, Any]]:
	if raw is None:
		return []
	if isinstance(raw, list):
		return raw
	if isinstance(raw, dict):
		data = raw.get("data") or raw.get("positions") or raw.get("result")
		if isinstance(data, list):
			return data
		if isinstance(data, dict):
			return [data]
		return [raw]
	return []


def fetch_snapshot(symbols: Iterable[str], creds: Dict[str, str]) -> DashboardState:
	state = DashboardState()
	try:
		state.account = get_account_assets(**creds, verbose=False)
	except Exception as exc:  # pragma: no cover - defensive
		state.error = f"Error al leer cuenta: {exc}"

	for symbol in symbols:
		try:
			state.prices[symbol] = get_ticker_price(**creds, symbol=symbol, verbose=False)
		except Exception as exc:  # pragma: no cover - defensive
			state.prices[symbol] = None
			state.error = f"Error al leer precio de {symbol}: {exc}"

		try:
			pos = get_positions(**creds, symbol=symbol, verbose=False)
			if pos:
				state.positions[symbol] = pos
		except Exception as exc:  # pragma: no cover - defensive
			state.error = f"Error al leer posiciones: {exc}"

	state.last_update = datetime.utcnow()
	return state


def build_account_panel(account: Optional[Dict[str, Any]]) -> Panel:
	table = Table(title="Cuenta", expand=True)
	table.add_column("Campo", style="cyan", no_wrap=True)
	table.add_column("Valor", style="bold")

	if not account:
		table.add_row("estado", "Sin datos. Revisa credenciales o conexion.")
		return Panel(table, title="Balance")

	data = account.get("data") if isinstance(account, dict) else None
	data = data or account

	table.add_row("total_equity", safe_float(data.get("total_equity", "--")))
	table.add_row("balance", safe_float(data.get("balance", "--")))
	table.add_row("unrealized_pnl", safe_float(data.get("unrealized_pnl", "--")))
	table.add_row("realised_pnl", safe_float(data.get("realised_pnl", "--")))

	coins = data.get("coins") if isinstance(data, dict) else None
	if coins:
		coins_table = Table(title="Activos", show_edge=False, show_header=True, expand=True)
		coins_table.add_column("Moneda", style="cyan", no_wrap=True)
		coins_table.add_column("Disponible", justify="right")
		coins_table.add_column("En uso", justify="right")
		for coin in coins:
			symbol = str(coin.get("coin", "?")).upper()
			avail = safe_float(coin.get("available", 0))
			frozen = safe_float(coin.get("frozen", 0))
			coins_table.add_row(symbol, avail, frozen)
		content = Group(table, coins_table)
		return Panel(content, title="Balance")

	return Panel(table, title="Balance")


def build_price_panel(prices: Dict[str, Optional[float]]) -> Panel:
	table = Table(title="Precios", expand=True)
	table.add_column("Simbolo", style="cyan", no_wrap=True)
	table.add_column("Ultimo", justify="right")
	for symbol, price in prices.items():
		table.add_row(symbol, safe_float(price) if price is not None else "--")
	return Panel(table, title="Mercado")


def build_positions_panel(positions: Dict[str, Any]) -> Panel:
	table = Table(title="Posiciones", expand=True)
	table.add_column("Simbolo", style="cyan", no_wrap=True)
	table.add_column("Side", style="magenta")
	table.add_column("Size", justify="right")
	table.add_column("Entry", justify="right")
	table.add_column("Mark", justify="right")
	table.add_column("PnL", justify="right")
	table.add_column("Lev", justify="right")

	any_rows = False
	for symbol, raw in positions.items():
		for pos in normalize_positions(raw):
			side = str(pos.get("holdSide") or pos.get("side") or "?")
			size = safe_float(pos.get("size") or pos.get("qty") or pos.get("positionMargin") or "--")
			entry = safe_float(pos.get("avgPrice") or pos.get("entryPrice") or pos.get("open_price") or "--")
			mark = safe_float(pos.get("markPrice") or pos.get("marketPrice") or pos.get("last") or "--")
			pnl = safe_float(pos.get("unrealizedPnl") or pos.get("pnl") or pos.get("unrealized_pnl") or "--")
			lev = safe_float(pos.get("leverage") or pos.get("leverage_ratio") or "--")
			table.add_row(symbol, side, size, entry, mark, pnl, lev)
			any_rows = True

	if not any_rows:
		table.add_row("-", "-", "-", "-", "-", "-", "-")

	return Panel(table, title="Posiciones")


def render_layout(state: DashboardState, symbols: Iterable[str], refresh_seconds: float = DEFAULT_REFRESH_SECONDS) -> Layout:
	layout = Layout()
	layout.split(
		Layout(name="header", size=3),
		Layout(name="body", ratio=1),
	)

	layout["body"].split_row(
		Layout(name="account", ratio=2),
		Layout(name="market", ratio=1),
		Layout(name="positions", ratio=2),
	)

	header = Text()
	header.append("LLQuant Monitor ", style="bold green")
	header.append("| Symbols: ")
	header.append(", ".join(symbols), style="cyan")
	header.append(" | Refresca cada ")
	header.append(f"{refresh_seconds}s", style="yellow")
	header.append(" | Ultima actualizacion ")
	header.append(state.last_update.strftime("%Y-%m-%d %H:%M:%S UTC"), style="magenta")
	if state.error:
		header.append(f" | {state.error}", style="bold red")

	layout["header"].update(Panel(Align.center(header)))
	layout["account"].update(build_account_panel(state.account))
	layout["market"].update(build_price_panel(state.prices))
	layout["positions"].update(build_positions_panel(state.positions))
	return layout


def main(refresh_seconds: float = DEFAULT_REFRESH_SECONDS) -> None:
	console = Console()
	symbols = load_symbols()
	creds = load_credentials()

	if creds is None:
		console.print("[bold red]Faltan credenciales.[/bold red] Exporta WEEX_API_KEY, WEEX_SECRET_KEY y WEEX_PASSPHRASE.")
		console.print("Ejemplo (PowerShell): $env:WEEX_API_KEY='xxx'")
		return

	console.print("Iniciando dashboard. Ctrl+C para salir.")
	with Live(render_layout(DashboardState(), symbols, refresh_seconds), refresh_per_second=4, console=console, screen=True) as live:
		try:
			while True:
				state = fetch_snapshot(symbols, creds)
				live.update(render_layout(state, symbols, refresh_seconds))
				time.sleep(refresh_seconds)
		except KeyboardInterrupt:
			console.print("Saliendo del monitor.")


if __name__ == "__main__":
	main()
