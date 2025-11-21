import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    fetch_price_history,
    get_financial_highlights,
    rebase_to_100,
    load_multi_prices,
    compute_portfolio_metrics,
    capm_analysis,
    rolling_beta_series,
    efficient_frontier_simulation,
)

st.set_page_config(page_title="Valuación & Portafolio", page_icon="💹", layout="wide")

# Sidebar navigation
st.sidebar.title("💼 Menú")
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
    st.title("📈 Consulta de Acciones")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker = st.text_input("Ticker (ej. AAPL, MSFT, TSLA)", value="AAPL").upper().strip()
    with c2:
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
    with c3:
        interval = st.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=0)

    if ticker:
        df = fetch_price_history(ticker, period=period, interval=interval)
        if df is None or df.empty:
            st.error("No se pudieron obtener datos del ticker. Verifica el símbolo o intenta con otro periodo/intervalo (p. ej. 1y y 1d).")
        else:
            # Candlestick chart
            st.subheader(f"Gráfica de velas: {ticker}")
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name=ticker,
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Precio",
                xaxis_rangeslider_visible=False,
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Highlights financieros
            st.subheader("Top 10 datos financieros relevantes")
            highlights = get_financial_highlights(ticker)
            st.dataframe(highlights, use_container_width=True)

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
                )
                line.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(line, use_container_width=True)
            else:
                st.info("No fue posible cargar datos para la comparación.")

# --- MÓDULO 2: Portafolio / Simulador de Compras ---
elif mod == "Portafolio / Simulador de Compras":
    st.title("🧮 Portafolio & Simulador de Compras")

    default_universe = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "KO",
    ]

    with st.expander("🧰 Ayuda rápida", expanded=False):
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
            )
            fig_line.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")
            st.subheader("Construcción de portafolio por **pesos**")
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

                st.plotly_chart(metrics["rebased_fig"], use_container_width=True)

            st.markdown("---")
            st.subheader("Simulador de **compras** por número de acciones")
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
                pie = px.pie(pie_df, names="Ticker", values="Peso", title="Distribución por valor")
                pie.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(pie, use_container_width=True)

                # Evolución hipotética (buy & hold)
                port_val = (prices[sel] * np.array(qtys)).sum(axis=1) + max(rem, 0.0)
                base = float(port_val.iloc[0])
                port_idx = port_val / base * 100.0
                bench_idx = rebase_to_100(prices[["^GSPC"]])["^GSPC"]
                comp = pd.DataFrame({"Portafolio (compras)": port_idx, "S&P 500": bench_idx})
                comp_fig = px.line(comp, x=comp.index, y=comp.columns, labels={"value": "Índice (100=Inicio)"})
                comp_fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(comp_fig, use_container_width=True)
    else:
        st.info("Selecciona al menos un ticker para continuar.")

# --- MÓDULO 3: Análisis CAPM ---
elif mod == "Análisis CAPM":
    st.title("📐 Análisis CAPM")

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
            scatter.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(scatter, use_container_width=True)

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
            sml_fig.add_trace(go.Scatter(x=betas, y=sml * 100, mode="lines", name="SML"))
            sml_fig.add_trace(go.Scatter(x=[b], y=[capm_er * 100], mode="markers", name=ticker, marker=dict(size=10)))
            sml_fig.update_layout(
                xaxis_title="Beta",
                yaxis_title="Rendimiento esperado anual (%)",
                height=460,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(sml_fig, use_container_width=True)

            # Beta rolling
            st.subheader("Beta móvil")
            roll = rolling_beta_series(r_df[["Ri", "Rm"]], window=60)
            if roll is not None and not roll.empty:
                roll_fig = px.line(roll, labels={"value": "Beta", "index": "Fecha"})
                roll_fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(roll_fig, use_container_width=True)

# --- MÓDULO 4: Portafolio Óptimo (Markowitz) ---
elif mod == "Portafolio Óptimo (Markowitz)":
    st.title("🎯 Portafolio Óptimo – Markowitz (simulación)")

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
            )
            # Puntos especiales
            for name, row in sim["specials"].items():
                dot.add_trace(go.Scatter(
                    x=[row["Vol"]], y=[row["Ret"]],
                    mode="markers+text", text=[name], textposition="top center",
                    marker=dict(size=12, symbol="star")
                ))
            dot.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(dot, use_container_width=True)

            st.markdown("### Pesos del **Tangency** (máx. Sharpe)")
            st.dataframe(sim["weights_tables"]["Tangency"], use_container_width=True)
            st.markdown("### Pesos del **Mínima Varianza**")
            st.dataframe(sim["weights_tables"]["Min Var"], use_container_width=True)

            st.markdown("### Series rebalanceadas (100=Inicio) para carteras especiales")
            series_fig = px.line(sim["rebased_series"], labels={"value": "Índice (100=Inicio)", "variable": "Cartera"})
            series_fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(series_fig, use_container_width=True)
