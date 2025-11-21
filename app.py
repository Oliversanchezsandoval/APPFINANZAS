import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    fetch_price_history,
    get_financial_highlights,
    get_company_profile,
    get_latest_news,
    rebase_to_100,
    load_multi_prices,
    compute_portfolio_metrics,
    capm_analysis,
    rolling_beta_series,
    efficient_frontier_simulation,
    compute_drawdown_episodes,
)

st.set_page_config(page_title="Valuación & Portafolio", layout="wide")

PRIMARY_COLOR = "#1d4ed8"
ACCENT_COLOR = "#ef4444"

COLOR_PALETTE = [
    PRIMARY_COLOR,
    "#0ea5e9",
    "#a855f7",
    "#22c55e",
    ACCENT_COLOR,
    "#f59e0b",
    "#6366f1",
    "#14b8a6",
]


def fig_to_png(fig):
    """Return PNG bytes for a Plotly figure using kaleido, or None if unavailable."""

    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception:
        return None


def render_export_controls(fig, data: pd.DataFrame | pd.Series, prefix: str):
    """Render paired download buttons for the active chart and its filtered data."""

    csv_bytes = None
    if data is not None:
        if isinstance(data, pd.Series):
            data = data.to_frame()
        try:
            csv_bytes = data.to_csv(index=True).encode("utf-8")
        except Exception:
            csv_bytes = None

    png_bytes = fig_to_png(fig)
    if not isinstance(png_bytes, (bytes, bytearray)):
        png_bytes = None

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Exportar gráfica (PNG)",
            data=png_bytes or b"",
            file_name=f"{prefix}.png",
            mime="image/png",
            disabled=png_bytes is None,
        )
    with b2:
        st.download_button(
            "Exportar datos filtrados (CSV)",
            data=csv_bytes or b"",
            file_name=f"{prefix}.csv",
            mime="text/csv",
            disabled=csv_bytes is None,
        )


def safe_update_layout(fig, **kwargs):
    """Update layout ignoring unsupported keys to avoid Plotly validation errors."""

    valid_keys = set(fig.layout._valid_props)
    filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    if filtered:
        fig.update_layout(**filtered)


def apply_elegant_layout(fig):
    safe_update_layout(
        fig,
        template="plotly_dark",
        plot_bgcolor="#0b1220",
        paper_bgcolor="#0b1220",
        font=dict(color="#e5e7eb", family="'Inter', 'Helvetica', sans-serif"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1f2937", zerolinecolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937", zerolinecolor="#1f2937")
    return fig


st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --accent-color: {ACCENT_COLOR};
    }}
    .stApp {{
        background: radial-gradient(circle at 20% 20%, #111827 0%, #0b1220 35%, #0b1220 100%);
        color: #e5e7eb;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Inter', 'Helvetica', sans-serif;
        letter-spacing: 0.2px;
        color: #f8fafc;
    }}
    .stSidebar, .stSidebar .stSelectbox, .stSidebar .stRadio {{
        background: #0b1220;
        color: #e5e7eb;
    }}
    .stSidebar h2, .stSidebar p, .stSidebar label {{
        color: #e5e7eb !important;
    }}
    .stButton>button, .stDownloadButton>button {{
        background-color: var(--primary-color);
        color: #f9fafb;
        border: 0;
        border-radius: 8px;
        padding: 0.6rem 1.1rem;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background-color: var(--accent-color);
        color: #fff;
    }}
    .stMetric label {{
        color: #cbd5e1;
    }}
    .stDataFrame, .stTable {{
        background: #0f172a;
        color: #e5e7eb;
    }}
    .stDataFrame th, .stDataFrame td, .stTable th, .stTable td {{
        color: #e5e7eb !important;
        background-color: #0f172a !important;
    }}
    .stAlert {{
        background-color: #111827;
        color: #e5e7eb;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("Menú principal")
mod = st.sidebar.radio(
    "Módulos",
    [
        "Consulta de Acciones",
        "Portafolio / Simulador de Compras",
        "Análisis CAPM",
        "Portafolio Óptimo (Markowitz)",
    ],
    index=0,
)

st.sidebar.caption("Datos de mercado vía yfinance. Gráficas con Plotly.")

# --- MÓDULO 1: Consulta de Acciones ---
if mod == "Consulta de Acciones":
    st.title("Consulta de Acciones")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker = st.text_input("Ticker (ej. AAPL, MSFT, TSLA)", value="AAPL").upper().strip()
    with c2:
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
    with c3:
        interval = st.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=0)

    dd_context = None

    if ticker:
        df = fetch_price_history(ticker, period=period, interval=interval)
        if df is None or df.empty:
            st.error("No se pudieron obtener datos del ticker. Verifica el símbolo o intenta con otro periodo/intervalo (p. ej. 1y y 1d).")
        else:
            # Highlights financieros
            st.subheader("Top 10 datos financieros relevantes")
            highlights = get_financial_highlights(ticker)
            st.dataframe(highlights, use_container_width=True)

            # Perfil y noticia más reciente
            st.subheader("Perfil y última noticia")
            profile = get_company_profile(ticker)
            news = get_latest_news(ticker)

            name = profile.get("name") or ticker
            logo = profile.get("logo_url")
            summary = profile.get("summary_short") or profile.get("summary")

            profile_card = st.container()
            with profile_card:
                info_col, news_col = st.columns([1.2, 1])

                with info_col:
                    meta_parts = []
                    if profile.get("sector"):
                        meta_parts.append(profile["sector"])
                    if profile.get("industry"):
                        meta_parts.append(profile["industry"])
                    meta = " · ".join(meta_parts) if meta_parts else None

                    if logo:
                        header_html = f"""
                        <div style="display:flex; align-items:center; gap:12px; padding:6px 0;">
                            <img src="{logo}" alt="Logo de {name}" style="width:64px; height:64px; object-fit:contain; background:#0f172a; border-radius:12px; padding:6px; border:1px solid #1f2937;" />
                            <div style="line-height:1.2;">
                                <div style="font-size:1.1rem; font-weight:700; color:#f8fafc;">{name}</div>
                                {f'<div style="color:#94a3b8; font-size:0.9rem;">{meta}</div>' if meta else ''}
                            </div>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                    else:
                        st.markdown(f"### {name}")
                        if meta:
                            st.caption(meta)

                    if profile.get("website"):
                        st.markdown(f"[Sitio web]({profile['website']})")

                    ceo_line = []
                    if profile.get("ceo"):
                        ceo_line.append(f"CEO: {profile['ceo']}")
                    if profile.get("employees"):
                        ceo_line.append(f"Empleados: {int(profile['employees']):,}".replace(",", "."))
                    if ceo_line:
                        st.caption(" · ".join(ceo_line))

                    st.markdown("**Descripción breve**")
                    if summary:
                        st.write(summary)
                    else:
                        fallback = []
                        if profile.get("sector"):
                            fallback.append(profile["sector"])
                        if profile.get("industry"):
                            fallback.append(profile["industry"])
                        if profile.get("ceo"):
                            fallback.append(f"CEO: {profile['ceo']}")
                        if fallback:
                            st.write(" · ".join(fallback))
                        else:
                            st.info("No se encontró una descripción breve para esta empresa.")

                with news_col:
                    st.markdown("**Yahoo Finance – Noticia reciente**")
                    if news:
                        title = news.get("title") or "Noticia reciente"
                        link = news.get("link")
                        publisher = news.get("publisher")
                        ts = news.get("published")
                        date_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notnull(ts) else None
                        summary_text = news.get("summary")

                        if publisher or date_str:
                            meta = " · ".join([p for p in [publisher, date_str] if p])
                            st.caption(meta)

                        if link:
                            st.markdown(f"**[{title}]({link})**")
                        else:
                            st.markdown(f"**{title}**")

                        if summary_text:
                            st.write(summary_text)
                        elif not link:
                            st.info("No se encontró el detalle de la nota.")
                    else:
                        st.info("No se encontraron noticias recientes para este ticker.")

            # Análisis de riesgo
            st.subheader("Riesgo de la acción")
            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            price_series = df[price_col].dropna()
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]

            returns = price_series.pct_change().dropna()
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]

            ann_map = {"1d": 252, "1wk": 52, "1mo": 12}
            ann_factor = ann_map.get(interval, 252)
            freq_label = {"1d": "diario", "1wk": "semanal", "1mo": "mensual"}.get(interval, "diario")

            if returns.empty:
                st.warning("No hay suficientes datos para calcular métricas de riesgo en este periodo.")
            else:
                ann_vol = returns.std() * np.sqrt(ann_factor)
                var_95 = returns.quantile(0.05)
                tail_losses = returns[returns <= var_95]
                cvar_95 = tail_losses.mean() if not tail_losses.empty else np.nan

                rolling_window = 20 if interval == "1d" else 12
                roll_vol = returns.rolling(rolling_window).std() * np.sqrt(ann_factor)

                m1, m2, m3 = st.columns(3)
                m1.metric("Volatilidad anualizada", f"{ann_vol*100:.2f}%")
                m2.metric(f"VaR 95% ({freq_label})", f"{var_95*100:.2f}%")
                m3.metric(f"CVaR 95% ({freq_label})", f"{cvar_95*100:.2f}%")

                roll_df = roll_vol.dropna().to_frame("Volatilidad")
                perf_curve = (1 + returns).cumprod()
                running_max = perf_curve.cummax()
                drawdown = (perf_curve / running_max - 1).rename("Drawdown")
                dd_context = {
                    "returns": returns,
                    "perf_curve": perf_curve,
                    "running_max": running_max,
                    "drawdown": drawdown,
                }

                t1, t2 = st.tabs(["Volatilidad móvil", "Caídas máximas"])

                with t1:
                    if roll_df.empty:
                        st.info("Aún no hay suficientes observaciones para mostrar la volatilidad móvil.")
                    else:
                        risk_fig = px.line(
                            roll_df,
                            labels={"index": "Fecha", "Volatilidad": "Volatilidad (anualizada)"},
                            color_discrete_sequence=[PRIMARY_COLOR],
                        )
                        safe_update_layout(
                            risk_fig,
                            height=420,
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        risk_fig = apply_elegant_layout(risk_fig)
                        st.plotly_chart(risk_fig, use_container_width=True)
                        render_export_controls(risk_fig, roll_df, f"{ticker}_volatilidad")

                with t2:
                    if drawdown.empty:
                        st.info("Sin suficientes datos para estimar drawdowns.")
                    else:
                        dd_fig = px.area(
                            drawdown.to_frame(),
                            labels={"index": "Fecha", "Drawdown": "Drawdown"},
                            color_discrete_sequence=[ACCENT_COLOR],
                        )
                        dd_fig.update_yaxes(tickformat=".0%", range=[drawdown.min(), 0])
                        safe_update_layout(
                            dd_fig,
                            height=420,
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        dd_fig = apply_elegant_layout(dd_fig)
                        st.plotly_chart(dd_fig, use_container_width=True)
                        render_export_controls(dd_fig, drawdown, f"{ticker}_drawdown")

            st.markdown("---")
            st.subheader("Drawdown & Recuperación")

            if not dd_context:
                st.info("No hay suficientes datos para analizar drawdowns y recuperaciones en este ticker.")
            else:
                drawdown = dd_context["drawdown"]
                perf_curve = dd_context["perf_curve"]
                running_max = dd_context["running_max"]

                dd_series, _, episodes = compute_drawdown_episodes(perf_curve)
                if dd_series is None or dd_series.empty:
                    st.info("No se pudieron calcular episodios de drawdown.")
                else:
                    max_depth = dd_series.min()
                    current_dd = dd_series.iloc[-1]
                    zeros = dd_series[dd_series >= -1e-9]
                    last_peak = zeros.index.max() if not zeros.empty else dd_series.index[0]
                    days_since_peak = (dd_series.index[-1] - last_peak).days if last_peak is not None else None

                    recovered_eps = [e for e in episodes if e.get("days_to_recover") is not None]
                    longest_recovery = max((e["days_to_recover"] for e in recovered_eps), default=None)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Drawdown máximo", f"{max_depth*100:.2f}%")
                    m2.metric("Drawdown actual", f"{current_dd*100:.2f}%")
                    if days_since_peak is not None:
                        m3.metric("Días desde máximo previo", f"{int(days_since_peak)} días")
                    else:
                        m3.metric("Días desde máximo previo", "-")
                    if longest_recovery is not None:
                        st.caption(f"Recuperación más larga: {int(longest_recovery)} días")

                    traj_df = pd.concat(
                        [
                            (perf_curve * 100).rename("Índice (100=Inicio)"),
                            (running_max * 100).rename("Máximo acumulado"),
                        ],
                        axis=1,
                    )
                    traj_fig = px.line(
                        traj_df,
                        labels={"index": "Fecha", "value": "Índice"},
                        color_discrete_sequence=[PRIMARY_COLOR, ACCENT_COLOR],
                    )
                    safe_update_layout(traj_fig, height=420, margin=dict(l=10, r=10, t=20, b=10))
                    traj_fig = apply_elegant_layout(traj_fig)
                    st.plotly_chart(traj_fig, use_container_width=True)
                    render_export_controls(traj_fig, traj_df, f"{ticker}_recuperacion")

                    episodes_df = pd.DataFrame(episodes)
                    if episodes_df.empty:
                        st.info("No se identificaron episodios de drawdown.")
                    else:
                        episodes_df = episodes_df.sort_values("depth")
                        episodes_df["peak"] = episodes_df["peak"].dt.date
                        episodes_df["trough"] = episodes_df["trough"].dt.date
                        episodes_df["recovery"] = episodes_df["recovery"].dt.date
                        episodes_df["depth_pct"] = episodes_df["depth"] * 100

                        nice_cols = {
                            "peak": "Inicio",
                            "trough": "Mínimo",
                            "recovery": "Recuperación",
                            "depth_pct": "Caída máx (%)",
                            "days_to_trough": "Días a mínimo",
                            "days_to_recover": "Días a recuperar",
                        }
                        view_df = episodes_df[nice_cols.keys()].rename(columns=nice_cols)
                        st.dataframe(
                            view_df,
                            use_container_width=True,
                            column_config={"Caída máx (%)": st.column_config.NumberColumn(format="%.2f")},
                        )

            # Comparación vs S&P 500
            st.subheader("Comparativa vs S&P 500 (rebalance a 100)")
            comp_df = load_multi_prices([ticker, "^GSPC"], period=period, interval="1d")
            if comp_df is not None and not comp_df.empty:
                rebased = rebase_to_100(comp_df)
                line = px.line(
                    rebased,
                    x=rebased.index,
                    y=rebased.columns,
                    labels={"value": "Índice (100=Inicio)", "variable": "Ticker"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
                safe_update_layout(line, height=420, margin=dict(l=10, r=10, t=10, b=10))
                line = apply_elegant_layout(line)
                st.plotly_chart(line, use_container_width=True)
                render_export_controls(line, rebased, f"{ticker}_comparativa")
            else:
                st.info("No fue posible cargar datos para la comparación.")

# --- MÓDULO 2: Portafolio / Simulador de Compras ---
elif mod == "Portafolio / Simulador de Compras":
    st.title("Portafolio & Simulador de Compras")

    default_universe = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "KO",
    ]

    with st.expander("Ayuda rápida", expanded=False):
        st.write(
            """
            1. Selecciona acciones para comparar.
            2. Ajusta fechas y horizonte de análisis.
            3. Asigna pesos o simula compras por número de acciones.
            4. Revisa métricas de portafolio y comparación vs S&P 500.
            """
        )

    c1, c2 = st.columns(2)
    with c1:
        sel = st.multiselect("Selecciona acciones", options=default_universe, default=["AAPL", "MSFT", "NVDA"])
    with c2:
        period = st.selectbox("Periodo", ["6mo", "1y", "2y", "5y"], index=1)

    if sel:
        prices = load_multi_prices(sel + ["^GSPC"], period=period, interval="1d")
        if prices is None or prices.empty:
            st.error("No se pudieron cargar precios para el universo seleccionado.")
        else:
            st.subheader("Evolución – Rebase a 100")
            rebased = rebase_to_100(prices[sel + ["^GSPC"]])
            fig_line = px.line(
                rebased,
                x=rebased.index,
                y=rebased.columns,
                labels={"value": "Índice (100=Inicio)", "variable": "Ticker"},
                color_discrete_sequence=COLOR_PALETTE,
            )
            safe_update_layout(fig_line, height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(apply_elegant_layout(fig_line), use_container_width=True)

            st.markdown("---")
            st.subheader("Construcción de portafolio por pesos")
            cols = st.columns(len(sel))
            weights = []
            for i, tk in enumerate(sel):
                with cols[i]:
                    w = st.number_input(
                        f"Peso {tk}",
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1 / len(sel), 4),
                        step=0.01,
                        key=f"w_{tk}",
                    )
                    weights.append(w)
            total_w = float(np.sum(weights))
            st.caption(f"Suma de pesos: {total_w:.2f} (debe ser 1.00)")
            if abs(total_w - 1.0) > 1e-6:
                st.warning("Ajusta los pesos para que sumen exactamente 1.00")
            else:
                metrics = compute_portfolio_metrics(prices[sel], weights, benchmark=prices["^GSPC"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rend. anualizado", f"{metrics['ann_return']*100:.2f}%")
                m2.metric("Volatilidad anualizada", f"{metrics['ann_vol']*100:.2f}%")
                m3.metric("Sharpe (rf~0)", f"{metrics['sharpe']:.2f}")
                m4.metric("Beta vs S&P 500", f"{metrics['beta']:.2f}")

                st.plotly_chart(apply_elegant_layout(metrics["rebased_fig"]), use_container_width=True)

            st.markdown("---")
            st.subheader("Simulador de compras por número de acciones")
            budget = st.number_input("Presupuesto (USD)", min_value=0.0, value=10000.0, step=100.0)

            latest = prices.iloc[-1][sel]
            buy_cols = st.columns(len(sel))
            qtys = []
            for i, tk in enumerate(sel):
                with buy_cols[i]:
                    st.caption(f"{tk}: precio actual ~ {latest[tk]:.2f}")
                    q = st.number_input(f"Cantidad {tk}", min_value=0, value=0, step=1, key=f"q_{tk}")
                    qtys.append(q)

            cost = float(np.sum(np.array(qtys) * latest.values))
            rem = budget - cost
            e1, e2 = st.columns(2)
            e1.metric("Costo total", f"${cost:,.2f}")
            e2.metric("Efectivo restante", f"${rem:,.2f}")

            if cost > 0:
                # Valor y ponderaciones implícitas
                weights_buy = (np.array(qtys) * latest.values) / cost
                pie_df = pd.DataFrame({"Ticker": sel, "Peso": weights_buy})
                pie = px.pie(
                    pie_df,
                    names="Ticker",
                    values="Peso",
                    title="Distribución por valor",
                    color_discrete_sequence=COLOR_PALETTE,
                )
                safe_update_layout(pie, height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(apply_elegant_layout(pie), use_container_width=True)

                # Evolución hipotética (buy & hold)
                port_val = (prices[sel] * np.array(qtys)).sum(axis=1) + max(rem, 0.0)
                base = float(port_val.iloc[0])
                port_idx = port_val / base * 100.0
                bench_idx = rebase_to_100(prices[["^GSPC"]])["^GSPC"]
                comp = pd.DataFrame({"Portafolio (compras)": port_idx, "S&P 500": bench_idx})
                comp_fig = px.line(
                    comp,
                    x=comp.index,
                    y=comp.columns,
                    labels={"value": "Índice (100=Inicio)"},
                    color_discrete_sequence=COLOR_PALETTE,
                )
                safe_update_layout(comp_fig, height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(apply_elegant_layout(comp_fig), use_container_width=True)
    else:
        st.info("Selecciona al menos un ticker para continuar.")

# --- MÓDULO 3: Análisis CAPM ---
elif mod == "Análisis CAPM":
    st.title("Análisis CAPM")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker = st.text_input("Ticker a analizar", value="AAPL").upper().strip()
    with c2:
        mkt = st.text_input("Índice de mercado (proxy)", value="^GSPC").upper().strip()
    with c3:
        period = st.selectbox("Periodo", ["6mo", "1y", "2y", "5y"], index=1)

    c4, c5 = st.columns(2)
    with c4:
        rf = st.number_input("Tasa libre de riesgo anual (%)", min_value=-5.0, max_value=20.0, value=3.0, step=0.1) / 100.0
    with c5:
        freq = st.selectbox("Frecuencia de rendimientos", ["Diaria", "Semanal", "Mensual"], index=0)

    if ticker:
        res = capm_analysis(ticker, mkt, period=period, freq=freq)
        if res["returns"] is None:
            st.error("No se pudieron descargar datos suficientes para el análisis.")
        else:
            r_df = res["returns"]
            st.caption(f"Observaciones usadas: {len(r_df)}")

            # Scatter y recta de regresión SIN statsmodels
            st.subheader("Rendimiento del activo vs. mercado")
            scatter = px.scatter(
                r_df,
                x="Rm",
                y="Ri",
                labels={"Rm": "Rendimiento mercado", "Ri": "Rendimiento activo"},
                color_discrete_sequence=[COLOR_PALETTE[0]],
            )
            a, b, r2, sigma_e = res["alpha"], res["beta"], res["r2"], res["sigma_e"]
            # Línea de regresión manual: y = a + b x
            x_min, x_max = float(r_df["Rm"].min()), float(r_df["Rm"].max())
            x_line = np.linspace(x_min, x_max, 100)
            y_line = a + b * x_line
            scatter.add_trace(
                go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Ri = {a:.4f} + {b:.4f}·Rm")
            )
            scatter.update_traces(marker=dict(size=6, opacity=0.7), selector=dict(mode="markers"))
            safe_update_layout(scatter, height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(apply_elegant_layout(scatter), use_container_width=True)

            # Métricas
            ann = 252 if freq == "Diaria" else (52 if freq == "Semanal" else 12)
            er_mkt = (1 + r_df["Rm"]).prod() ** (ann / len(r_df)) - 1
            capm_er = rf + b * (er_mkt - rf)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Beta", f"{b:.4f}")
            m2.metric("Alfa (intercepto)", f"{a*100:.2f}%")
            m3.metric("R²", f"{r2:.3f}")
            m4.metric("σ idiosincrática", f"{sigma_e*100:.2f}%")

            # SML
            st.subheader("Security Market Line (SML)")
            betas = np.linspace(0, 2, 50)
            sml = rf + (er_mkt - rf) * betas
            sml_fig = go.Figure()
            sml_fig.add_trace(
                go.Scatter(
                    x=betas,
                    y=sml * 100,
                    mode="lines",
                    name="SML",
                    line=dict(color=COLOR_PALETTE[0], width=3),
                )
            )
            sml_fig.add_trace(
                go.Scatter(
                    x=[b],
                    y=[capm_er * 100],
                    mode="markers",
                    name=ticker,
                    marker=dict(size=10, color=COLOR_PALETTE[4]),
                )
            )
            safe_update_layout(
                sml_fig,
                xaxis_title="Beta",
                yaxis_title="Rendimiento esperado anual (%)",
                height=460,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(apply_elegant_layout(sml_fig), use_container_width=True)

            # Beta rolling
            st.subheader("Beta móvil")
            roll = rolling_beta_series(r_df[["Ri", "Rm"]], window=60)
            if roll is not None and not roll.empty:
                roll_fig = px.line(
                    roll,
                    labels={"value": "Beta", "index": "Fecha"},
                    color_discrete_sequence=[COLOR_PALETTE[1]],
                )
                safe_update_layout(roll_fig, height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(apply_elegant_layout(roll_fig), use_container_width=True)

# --- MÓDULO 4: Portafolio Óptimo (Markowitz) ---
elif mod == "Portafolio Óptimo (Markowitz)":
    st.title("Portafolio Óptimo – Markowitz (simulación)")

    default_universe = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "KO"]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel = st.multiselect("Acciones", options=default_universe, default=["AAPL", "MSFT", "NVDA", "AMZN"])
    with c2:
        period = st.selectbox("Periodo", ["6mo", "1y", "2y", "5y"], index=1)
    with c3:
        rf = st.number_input("Tasa libre de riesgo anual (%)", min_value=-5.0, max_value=20.0, value=3.0, step=0.1) / 100.0

    c4, c5 = st.columns(2)
    with c4:
        allow_short = st.checkbox("Permitir cortos (short selling)", value=False)
    with c5:
        nport = st.slider("N° de portafolios simulados", min_value=2000, max_value=40000, value=15000, step=1000)

    if sel:
        sim = efficient_frontier_simulation(sel, period=period, rf=rf, n_portfolios=nport, allow_short=allow_short)
        if sim["prices"] is None:
            st.error("No se pudieron descargar precios. Intenta con otro periodo.")
        else:
            # Dispersión riesgo-rendimiento coloreada por Sharpe
            st.subheader("Frontera eficiente (simulación Monte Carlo)")
            dot = px.scatter(
                sim["all_stats"],
                x="Vol",
                y="Ret",
                color="Sharpe",
                hover_data=sim["all_stats"].columns,
                labels={"Vol": "Volatilidad anual", "Ret": "Rendimiento anual"},
                color_continuous_scale="Tealrose",
            )
            # Puntos especiales
            for name, row in sim["specials"].items():
                dot.add_trace(go.Scatter(
                    x=[row["Vol"]], y=[row["Ret"]],
                    mode="markers+text", text=[name], textposition="top center",
                    marker=dict(size=12, symbol="star")
                ))
            safe_update_layout(dot, height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(apply_elegant_layout(dot), use_container_width=True)

            st.markdown("### Pesos del **Tangency** (máx. Sharpe)")
            st.dataframe(sim["weights_tables"]["Tangency"], use_container_width=True)
            st.markdown("### Pesos del **Mínima Varianza**")
            st.dataframe(sim["weights_tables"]["Min Var"], use_container_width=True)

            st.markdown("### Series rebalanceadas (100=Inicio) para carteras especiales")
            series_fig = px.line(
                sim["rebased_series"],
                labels={"value": "Índice (100=Inicio)", "variable": "Cartera"},
                color_discrete_sequence=COLOR_PALETTE,
            )
            safe_update_layout(series_fig, height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(apply_elegant_layout(series_fig), use_container_width=True)
