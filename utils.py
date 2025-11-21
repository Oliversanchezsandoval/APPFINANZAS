import html
import re
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px


# =========================
# Utilidades de respaldo
# =========================


def _estimate_period_rows(period: str, interval: str) -> int:
    """Devuelve un número aproximado de observaciones según periodo/intervalo."""

    period_map = {
        "1mo": 22,
        "3mo": 66,
        "6mo": 126,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
        "10y": 2520,
        "max": 2520,
    }
    base = period_map.get(period, 252)
    if interval == "1wk":
        return max(10, base // 5)
    if interval == "1mo":
        return max(6, base // 22)
    return max(20, base)


def _generate_placeholder_history(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    """Genera una serie OHLC sintética para mantener la app operativa sin datos reales."""

    n = _estimate_period_rows(period, interval)
    freq = {"1d": "B", "1wk": "W-FRI", "1mo": "M"}.get(interval, "B")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq=freq)

    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    rets = rng.normal(0.0005, 0.02 if interval == "1d" else 0.04, size=n)
    price = 100 * (1 + pd.Series(rets, index=idx)).cumprod()

    noise = rng.normal(0.0, 0.008, size=n)
    high = price * (1 + np.abs(noise))
    low = price * (1 - np.abs(noise))

    df = pd.DataFrame(
        {
            "Open": price.shift(1).fillna(price.iloc[0]),
            "High": high,
            "Low": low,
            "Close": price,
            "Adj Close": price,
            "Volume": rng.integers(500_000, 5_000_000, size=n),
        },
        index=idx,
    )
    df.attrs["placeholder"] = True
    return df


def _placeholder_profile(ticker: str) -> dict:
    """Fallback de perfil para entornos sin conexión o respuestas vacías de Yahoo."""

    clean = (ticker or "").upper() or "TCKR"
    return {
        "name": clean,
        "summary": "Perfil no disponible temporalmente.",
        "summary_short": "Perfil no disponible temporalmente.",
        "sector": None,
        "industry": None,
        "website": None,
        "logo_url": None,
        "ceo": None,
        "employees": None,
        "placeholder": True,
    }

# =========================
# Descarga de datos robusta
# =========================

@st.cache_data(show_spinner=False)
def fetch_price_history(ticker: str, period="1y", interval="1d"):
    """
    Descarga robusta con reintentos/fallback y normalización de columnas/índice.
    Evita falsos vacíos de yfinance y problemas de MultiIndex/zonas horarias.
    """
    try:
        tk = (ticker or "").upper().strip()
        if not tk:
            return None

        attempts = [(period, interval), ("1y", "1d"), ("6mo", "1d"), ("1mo", "1d")]

        for per, inter in attempts:
            try:
                df = yf.download(
                    tk, period=per, interval=inter,
                    auto_adjust=False, progress=False, threads=False
                )
                if df is None or df.empty:
                    continue

                needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                for c in needed:
                    if c not in df.columns:
                        if c == "Adj Close" and "Close" in df.columns:
                            df["Adj Close"] = df["Close"]
                        else:
                            df[c] = np.nan

                df = df.sort_index()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df
            except Exception:
                continue
    except Exception:
        pass

    # Fallback sintético para mantener la app operativa
    return _generate_placeholder_history(tk or "PX", period=period, interval=interval)

@st.cache_data(show_spinner=False)
def load_multi_prices(tickers, period="1y", interval="1d"):
    """
    Devuelve un DataFrame de precios 'Adj Close' (o 'Close' si no hay) para 1+ tickers.
    Soporta salida con MultiIndex de yfinance y asegura índice ordenado.
    """
    try:
        data = yf.download(
            tickers, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=False
        )
        if data is not None and not data.empty:
            adj = None
            if isinstance(data.columns, pd.MultiIndex):
                if "Adj Close" in data.columns.get_level_values(0):
                    adj = data["Adj Close"]
                elif "Adj Close" in data.columns.get_level_values(-1):
                    adj = data.xs("Adj Close", axis=1, level=-1)
                elif "Close" in data.columns.get_level_values(0):
                    adj = data["Close"]
                elif "Close" in data.columns.get_level_values(-1):
                    adj = data.xs("Close", axis=1, level=-1, drop_level=True)
            else:
                if "Adj Close" in data.columns:
                    adj = data["Adj Close"]
                elif "Close" in data.columns:
                    adj = data["Close"]

            if adj is not None:
                if isinstance(adj, pd.Series):
                    name = tickers if isinstance(tickers, str) else (tickers[0] if tickers else "PX")
                    adj = adj.to_frame(name=name)

                adj = adj.dropna(how="all").sort_index()
                adj.index = pd.to_datetime(adj.index).tz_localize(None)
                if not adj.empty:
                    return adj
    except Exception:
        pass

    # Si falla la descarga, generar trayectorias sintéticas para todos los tickers solicitados
    tick_list = tickers if isinstance(tickers, (list, tuple, pd.Index)) else [tickers]
    synthetic = []
    for tk in tick_list:
        df = _generate_placeholder_history(tk, period=period, interval=interval)
        synthetic.append(df["Adj Close"].rename(tk))
    adj = pd.concat(synthetic, axis=1)
    adj.attrs["placeholder"] = True
    return adj

@st.cache_data(show_spinner=False)
def get_financial_highlights(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        fast = getattr(t, "fast_info", {}) or {}

        def get_attr(obj, key, default=None):
            try:
                return obj.get(key, default)
            except Exception:
                return default

        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        last_price = get_attr(fast, "last_price", None)
        if last_price is None:
            px_df = fetch_price_history(ticker, period="1mo", interval="1d")
            if px_df is not None and not px_df.empty:
                last_price = float(px_df["Close"].iloc[-1])

        metrics = {
            "Precio": last_price,
            "Market Cap": get_attr(fast, "market_cap", info.get("marketCap")),
            "P/E (TTM)": info.get("trailingPE"),
            "EPS (TTM)": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
            "52w Alto": info.get("fiftyTwoWeekHigh"),
            "52w Bajo": info.get("fiftyTwoWeekLow"),
            "Ingresos (ttm)": info.get("totalRevenue"),
            "Utilidad neta (ttm)": info.get("netIncomeToCommon"),
        }
    except Exception:
        metrics = {
            "Precio": None,
            "Market Cap": None,
            "P/E (TTM)": None,
            "EPS (TTM)": None,
            "Dividend Yield": None,
            "Beta": None,
            "52w Alto": None,
            "52w Bajo": None,
            "Ingresos (ttm)": None,
            "Utilidad neta (ttm)": None,
        }

    df = pd.DataFrame({"Métrica": list(metrics.keys()), "Valor": list(metrics.values())})
    return df


def _shorten_text(text: str | None, max_chars: int = 180) -> str | None:
    if not text:
        return None
    clean = str(text).strip()
    if len(clean) <= max_chars:
        return clean
    snippet = clean[:max_chars].rsplit(" ", 1)[0]
    return snippet + "…"


def _normalize_news_field(value) -> str | None:
    """Normalize news text fields to clean, plain strings."""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        value = " ".join([str(v) for v in value if v])
    elif isinstance(value, dict):
        # Prefer common text-bearing keys if a dict is provided
        for key in ("content", "body", "summary", "description", "title"):
            if key in value and value[key]:
                value = value[key]
                break
        else:
            value = str(value)

    text = html.unescape(str(value))
    # Remove HTML tags and collapse whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def compute_drawdown_episodes(perf_series: pd.Series):
    """Return drawdown, running max and detailed drawdown/recovery episodes."""

    if perf_series is None:
        return None, None, []

    series = perf_series.dropna()
    if series.empty:
        return series, series, []

    running_max = series.cummax()
    drawdown = series / running_max - 1

    episodes = []
    in_dd = False
    start = trough = None
    trough_val = 0.0

    for date, dd in drawdown.items():
        if not in_dd and dd < 0:
            in_dd = True
            start = date
            trough = date
            trough_val = float(dd)

        if in_dd:
            if dd < trough_val:
                trough_val = float(dd)
                trough = date

            if dd >= 0:
                peak_ts = pd.to_datetime(start) if start is not None else pd.NaT
                trough_ts = pd.to_datetime(trough) if trough is not None else pd.NaT
                rec_ts = pd.to_datetime(date)
                episodes.append(
                    {
                        "peak": peak_ts,
                        "trough": trough_ts,
                        "recovery": rec_ts,
                        "depth": trough_val,
                        "days_to_trough": (trough_ts - peak_ts).days if pd.notnull(trough_ts) else None,
                        "days_to_recover": (rec_ts - peak_ts).days if pd.notnull(rec_ts) else None,
                    }
                )
                in_dd = False
                start = trough = None
                trough_val = 0.0

    if in_dd:
        peak_ts = pd.to_datetime(start) if start is not None else pd.NaT
        trough_ts = pd.to_datetime(trough) if trough is not None else pd.NaT
        episodes.append(
            {
                "peak": peak_ts,
                "trough": trough_ts,
                "recovery": None,
                "depth": trough_val,
                "days_to_trough": (trough_ts - peak_ts).days if pd.notnull(trough_ts) else None,
                "days_to_recover": None,
            }
        )

    return drawdown, running_max, episodes


@st.cache_data(show_spinner=False)
def get_company_profile(ticker: str) -> dict:
    """Return basic company profile details, short description and logo for a ticker."""
    try:
        t = yf.Ticker(ticker)

        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        summary = info.get("longBusinessSummary") or info.get("longSummary")
        # CEO detection prioritizes officers with a CEO title, otherwise falls back to info fields
        ceo_name = None
        officers = info.get("companyOfficers") or []
        for officer in officers:
            title = str(officer.get("title") or "").lower()
            if "ceo" in title or "chief executive" in title:
                ceo_name = officer.get("name")
                break
        ceo_name = ceo_name or info.get("companyCEO") or info.get("ceo")

        profile = {
            "name": info.get("shortName") or info.get("longName") or ticker,
            "summary": summary,
            "summary_short": _shorten_text(summary, 160),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "logo_url": info.get("logo_url") or info.get("logoUrl"),
            "ceo": ceo_name,
            "employees": info.get("fullTimeEmployees"),
            "placeholder": False,
        }

        # Si todo está vacío, devolver un placeholder legible en lugar de None general
        if all(v is None for k, v in profile.items() if k not in ("name", "placeholder")):
            return _placeholder_profile(ticker)
        return profile
    except Exception:
        return _placeholder_profile(ticker)


@st.cache_data(show_spinner=False)
def get_latest_news(ticker: str) -> dict | None:
    """Return the latest available news item for a ticker from Yahoo Finance."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []

    if not items:
        return None

    first = items[0] or {}

    # Some providers include a summary/description; keep the richest available
    summary = _normalize_news_field(
        first.get("summary")
        or first.get("content")
        or first.get("description")
        or first.get("body")
    )
    return {
        "title": _normalize_news_field(first.get("title")) or "Noticia reciente",
        "link": first.get("link"),
        "publisher": first.get("publisher"),
        "published": pd.to_datetime(first.get("providerPublishTime"), unit="s", errors="coerce"),
        "summary": summary or "No se encontró el cuerpo de la nota.",
    }


def _shorten_text(text: str | None, max_chars: int = 180) -> str | None:
    if not text:
        return None
    clean = str(text).strip()
    if len(clean) <= max_chars:
        return clean
    snippet = clean[:max_chars].rsplit(" ", 1)[0]
    return snippet + "…"


def _normalize_news_field(value) -> str | None:
    """Normalize news text fields to clean, plain strings."""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        value = " ".join([str(v) for v in value if v])
    elif isinstance(value, dict):
        # Prefer common text-bearing keys if a dict is provided
        for key in ("content", "body", "summary", "description", "title"):
            if key in value and value[key]:
                value = value[key]
                break
        else:
            value = str(value)

    text = html.unescape(str(value))
    # Remove HTML tags and collapse whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def compute_drawdown_episodes(perf_series: pd.Series):
    """Return drawdown, running max and detailed drawdown/recovery episodes."""

    if perf_series is None:
        return None, None, []

    series = perf_series.dropna()
    if series.empty:
        return series, series, []

    running_max = series.cummax()
    drawdown = series / running_max - 1

    episodes = []
    in_dd = False
    start = trough = None
    trough_val = 0.0

    for date, dd in drawdown.items():
        if not in_dd and dd < 0:
            in_dd = True
            start = date
            trough = date
            trough_val = float(dd)

        if in_dd:
            if dd < trough_val:
                trough_val = float(dd)
                trough = date

            if dd >= 0:
                peak_ts = pd.to_datetime(start) if start is not None else pd.NaT
                trough_ts = pd.to_datetime(trough) if trough is not None else pd.NaT
                rec_ts = pd.to_datetime(date)
                episodes.append(
                    {
                        "peak": peak_ts,
                        "trough": trough_ts,
                        "recovery": rec_ts,
                        "depth": trough_val,
                        "days_to_trough": (trough_ts - peak_ts).days if pd.notnull(trough_ts) else None,
                        "days_to_recover": (rec_ts - peak_ts).days if pd.notnull(rec_ts) else None,
                    }
                )
                in_dd = False
                start = trough = None
                trough_val = 0.0

    if in_dd:
        peak_ts = pd.to_datetime(start) if start is not None else pd.NaT
        trough_ts = pd.to_datetime(trough) if trough is not None else pd.NaT
        episodes.append(
            {
                "peak": peak_ts,
                "trough": trough_ts,
                "recovery": None,
                "depth": trough_val,
                "days_to_trough": (trough_ts - peak_ts).days if pd.notnull(trough_ts) else None,
                "days_to_recover": None,
            }
        )

    return drawdown, running_max, episodes


@st.cache_data(show_spinner=False)
def get_company_profile(ticker: str) -> dict:
    """Return basic company profile details, short description and logo for a ticker."""
    t = yf.Ticker(ticker)

    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    summary = info.get("longBusinessSummary") or info.get("longSummary")
    # CEO detection prioritizes officers with a CEO title, otherwise falls back to info fields
    ceo_name = None
    officers = info.get("companyOfficers") or []
    for officer in officers:
        title = str(officer.get("title") or "").lower()
        if "ceo" in title or "chief executive" in title:
            ceo_name = officer.get("name")
            break
    ceo_name = ceo_name or info.get("companyCEO") or info.get("ceo")

    return {
        "name": info.get("shortName") or info.get("longName") or ticker,
        "summary": summary,
        "summary_short": _shorten_text(summary, 160),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "logo_url": info.get("logo_url") or info.get("logoUrl"),
        "ceo": ceo_name,
        "employees": info.get("fullTimeEmployees"),
    }


@st.cache_data(show_spinner=False)
def get_latest_news(ticker: str) -> dict | None:
    """Return the latest available news item for a ticker from Yahoo Finance."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return None

    if not items:
        return None

    first = items[0] or {}

    # Some providers include a summary/description; keep the richest available
    summary = _normalize_news_field(
        first.get("summary")
        or first.get("content")
        or first.get("description")
        or first.get("body")
    )
    return {
        "title": _normalize_news_field(first.get("title")),
        "link": first.get("link"),
        "publisher": first.get("publisher"),
        "published": pd.to_datetime(first.get("providerPublishTime"), unit="s", errors="coerce"),
        "summary": summary,
    }

def rebase_to_100(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(how="all")
    base = df.iloc[0]
    return df.divide(base) * 100.0

# =========================
# Métricas y portafolio
# =========================

def _ann_factor_from_freq(freq: str) -> int:
    if freq.lower().startswith("diar"):
        return 252
    if freq.lower().startswith("seman"):
        return 52
    if freq.lower().startswith("mens"):
        return 12
    return 252

def compute_portfolio_metrics(price_df: pd.DataFrame, weights, benchmark: pd.Series):
    rets = price_df.pct_change().dropna()
    w = np.array(weights, dtype=float)
    port_ret = (rets * w).sum(axis=1)

    ann_factor = 252
    if len(port_ret) == 0:
        ann_return = np.nan
        ann_vol = np.nan
        sharpe = np.nan
    else:
        ann_return = (1 + port_ret).prod() ** (ann_factor / len(port_ret)) - 1
        ann_vol = port_ret.std() * np.sqrt(ann_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    bench_ret = benchmark.pct_change().dropna()
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if aligned.shape[0] > 2:
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var_b = np.var(aligned.iloc[:, 1])
        beta = cov / var_b if var_b > 0 else np.nan
    else:
        beta = np.nan

    if len(port_ret) > 0:
        port_curve = (1 + port_ret).cumprod()
        port_idx = port_curve / port_curve.iloc[0] * 100.0
    else:
        port_idx = pd.Series(dtype=float)

    bench_idx = (1 + bench_ret).cumprod()
    if len(bench_idx) > 0:
        bench_idx = bench_idx / bench_idx.iloc[0] * 100.0

    comp = pd.DataFrame({"Portafolio": port_idx, "S&P 500": bench_idx}).dropna(how="all")
    fig = px.line(comp, x=comp.index, y=comp.columns, labels={"value": "Índice (100=Inicio)", "variable": "Serie"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))

    return {
        "ann_return": float(ann_return) if ann_return == ann_return else np.nan,
        "ann_vol": float(ann_vol) if ann_vol == ann_vol else np.nan,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "beta": float(beta) if beta == beta else np.nan,
        "rebased_fig": fig,
    }

# =========================
# CAPM
# =========================

@st.cache_data(show_spinner=False)
def capm_analysis(ticker: str, market: str, period="1y", freq="Diaria"):
    """
    Devuelve dict con:
      - returns: DataFrame con columnas Ri, Rm
      - alpha, beta, r2, sigma_e
    """
    prices_i = load_multi_prices([ticker], period=period, interval="1d")
    prices_m = load_multi_prices([market], period=period, interval="1d")
    placeholder = bool(getattr(prices_i, "attrs", {}).get("placeholder") or getattr(prices_m, "attrs", {}).get("placeholder"))
    if prices_i is None or prices_m is None:
        return {"returns": None, "placeholder": placeholder}

    df = pd.concat([prices_i[ticker], prices_m[market]], axis=1).dropna()
    df.columns = ["Pi", "Pm"]

    if freq.lower().startswith("seman"):
        df = df.resample("W-FRI").last()
    elif freq.lower().startswith("mens"):
        df = df.resample("M").last()

    ri = df["Pi"].pct_change().dropna()
    rm = df["Pm"].pct_change().dropna()
    aligned = pd.concat([ri, rm], axis=1).dropna()
    aligned.columns = ["Ri", "Rm"]
    if aligned.empty or len(aligned) < 20:
        return {"returns": None, "placeholder": placeholder}

    cov = np.cov(aligned["Ri"], aligned["Rm"])[0, 1]
    var_m = np.var(aligned["Rm"])
    beta = cov / var_m if var_m > 0 else np.nan
    alpha = aligned["Ri"].mean() - beta * aligned["Rm"].mean()

    corr = np.corrcoef(aligned["Ri"], aligned["Rm"])[0, 1]
    r2 = corr ** 2

    resid = aligned["Ri"] - (alpha + beta * aligned["Rm"])
    sigma_e = resid.std() * np.sqrt(_ann_factor_from_freq(freq))  # anual

    return {
        "returns": aligned,
        "alpha": float(alpha),
        "beta": float(beta),
        "r2": float(r2),
        "sigma_e": float(sigma_e),
        "placeholder": placeholder,
    }

def rolling_beta_series(ri_rm: pd.DataFrame, window=60) -> pd.DataFrame:
    if ri_rm is None or ri_rm.empty or "Ri" not in ri_rm or "Rm" not in ri_rm:
        return None
    cov_roll = ri_rm["Ri"].rolling(window).cov(ri_rm["Rm"])
    var_roll = ri_rm["Rm"].rolling(window).var()
    beta_roll = cov_roll / var_roll
    return beta_roll.dropna().to_frame("Beta")

# =========================
# Markowitz – Simulación
# =========================

def _annualized_stats(prices: pd.DataFrame):
    rets = prices.pct_change().dropna()
    mu_daily = rets.mean()
    cov_daily = rets.cov()
    mu = (1 + mu_daily) ** 252 - 1
    cov = cov_daily * 252
    return mu, cov

def _portfolio_stats(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
    sharpe = (ret - rf) / vol if vol > 0 else -np.inf
    return ret, vol, sharpe

@st.cache_data(show_spinner=False)
def efficient_frontier_simulation(tickers, period="1y", rf=0.03, n_portfolios=15000, allow_short=False):
    prices = load_multi_prices(tickers, period=period, interval="1d")
    placeholder = bool(getattr(prices, "attrs", {}).get("placeholder"))
    if prices is None or prices.empty:
        return {"prices": None, "placeholder": placeholder}

    mu, cov = _annualized_stats(prices)
    n = len(tickers)

    rng = np.random.default_rng(42)
    weights = []
    stats = []

    for _ in range(int(n_portfolios)):
        if allow_short:
            w = rng.normal(0, 1, n)
            w = w / np.sum(np.abs(w))
            w = w / np.sum(w)
        else:
            w = rng.random(n)
            w = w / w.sum()

        r, v, s = _portfolio_stats(w, mu, cov, rf)
        weights.append(w)
        stats.append((r, v, s))

    stats_df = pd.DataFrame(stats, columns=["Ret", "Vol", "Sharpe"])
    stats_df["ID"] = np.arange(len(stats_df))
    all_stats = stats_df

    idx_tan = all_stats["Sharpe"].idxmax()
    idx_mv = all_stats["Vol"].idxmin()
    w_tan = weights[idx_tan]
    w_mv = weights[idx_mv]

    specials = {
        "Tangency": all_stats.loc[idx_tan],
        "Min Var": all_stats.loc[idx_mv],
    }

    wt_df = pd.DataFrame({"Ticker": tickers, "Peso Tangency": w_tan, "Peso Min Var": w_mv}).set_index("Ticker")
    weights_tables = {
        "Tangency": wt_df[["Peso Tangency"]],
        "Min Var": wt_df[["Peso Min Var"]],
    }

    rets = prices.pct_change().dropna()
    series = {}
    if len(rets) > 0:
        for name, w in [("Tangency", w_tan), ("Min Var", w_mv)]:
            curve = (1 + (rets @ w)).cumprod()
            series[name] = curve / curve.iloc[0] * 100.0
    rebased_series = pd.DataFrame(series)

    return {
        "prices": prices,
        "all_stats": all_stats,
        "specials": specials,
        "weights_tables": weights_tables,
        "rebased_series": rebased_series,
        "placeholder": placeholder,
    }
