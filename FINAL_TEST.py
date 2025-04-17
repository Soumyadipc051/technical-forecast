import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, request
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Sample Nifty 500 list (add full list as needed)
nifty_500 = ( "ABB.NS", "ACC.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
    "ALKEM.NS", "AMARAJABAT.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS",
    "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS", "AUROPHARMA.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS",
    "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BASF.NS",
    "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS",
    "BHEL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS",
    "CANBK.NS", "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS",
    "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS",
    "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS",
    "DEEPAKNTR.NS", "DELHIVERY.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
    "FINCABLES.NS", "GAIL.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFC.NS", "HDFCAMC.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS",
    "HONAUT.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS",
    "IDFC.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS",
    "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS",
    "INFY.NS", "INTELLECT.NS", "IOB.NS", "IOC.NS", "IPCALAB.NS",
    "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", "JSWSTEEL.NS",
    "JUBLFOOD.NS", "KOTAKBANK.NS", "L&TFH.NS", "LALPATHLAB.NS", "LAURUSLABS.NS",
    "LICHSGFIN.NS", "LT.NS", "LTIM.NS", "LTTS.NS", "LUPIN.NS",
    "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS",
    "MAZDOCK.NS", "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS",
    "MGL.NS", "MINDTREE.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS",
    "NAM-INDIA.NS", "NATIONALUM.NS", "NAUKRI.NS", "NAVINFLUOR.NS", "NBCC.NS",
    "NCC.NS", "NHPC.NS", "NIACL.NS", "NLCINDIA.NS", "NMDC.NS",
    "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
    "PEL.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "PRAJIND.NS", "PRESTIGE.NS",
    "PVRINOX.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS",
    "RELAXO.NS", "RELIANCE.NS", "RENUKA.NS", "ROUTE.NS", "SAIL.NS",
    "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS",
    "SIEMENS.NS", "SJVN.NS", "SKFINDIA.NS", "SRF.NS", "SRTRANSFIN.NS",
    "SUNPHARMA.NS", "SUNDARMFIN.NS", "SUNTV.NS", "SUPREMEIND.NS", "SUZLON.NS",
    "SYMPHONY.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS",
    "ULTRACEMCO.NS", "UNIONBANK.NS", "UPL.NS", "VBL.NS", "VEDL.NS",
    "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
    
)

# --- Technical Indicator Functions ---

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short=12, long=26, signal=9):
    exp1 = prices.ewm(span=short, adjust=False).mean()
    exp2 = prices.ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return ma, upper, lower

# --- Flask Route ---
@app.route("/", methods=["GET"])
def index():
    stock = request.args.get("stock", "RELIANCE.NS")
    interval = request.args.get("interval", "1d")

    try:
        data = yf.download(tickers=stock, period="1y", interval=interval, progress=False)
        data = data[['Open', 'High', 'Low', 'Close']].dropna()
        data.index = pd.to_datetime(data.index)

        try:
            ticker_obj = yf.Ticker(stock)
            current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
        except:
            current_price = "N/A"

        # --- Forecasting with VAR ---
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

        model = VAR(scaled_data)
        results = model.fit(maxlags=5, ic='aic')
        forecast_input = scaled_data.values[-5:]
        forecast_scaled = results.forecast(y=forecast_input, steps=1)
        forecast = scaler.inverse_transform(forecast_scaled)
        forecast_dict = dict(zip(['Open', 'High', 'Low', 'Close'], forecast[0]))
        result = {k: round(v, 2) for k, v in forecast_dict.items()}

        # --- Forecast Bar Chart ---
        compare_df = pd.DataFrame([data.iloc[-1].values, forecast[0]],
                                  columns=['Open', 'High', 'Low', 'Close'],
                                  index=['Actual', 'Predicted'])
        compare_df.T.plot(kind='bar', title=f'{stock} - Next Day OHLC Prediction', figsize=(6, 4), color=['skyblue', 'orange'])
        plt.ylabel('Price')
        plt.tight_layout()
        bar_img = io.BytesIO()
        plt.savefig(bar_img, format='png')
        plt.close()
        bar_img.seek(0)
        bar_base64 = base64.b64encode(bar_img.read()).decode()

        # --- Price Trend Chart ---
        data['Close'].plot(title=f'{stock} - 1 Year Price Trend ({interval})', figsize=(8, 4), color='lightgreen')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.tight_layout()
        line_img = io.BytesIO()
        plt.savefig(line_img, format='png')
        plt.close()
        line_img.seek(0)
        line_base64 = base64.b64encode(line_img.read()).decode()

        # --- RSI Plot ---
        rsi = calculate_rsi(data['Close'])
        plt.figure(figsize=(8, 4))
        plt.plot(rsi, label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title(f'{stock} - RSI (14-period)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.tight_layout()
        rsi_img = io.BytesIO()
        plt.savefig(rsi_img, format='png')
        plt.close()
        rsi_img.seek(0)
        rsi_base64 = base64.b64encode(rsi_img.read()).decode()

        # --- MACD Plot ---
        macd, signal_line = calculate_macd(data['Close'])
        plt.figure(figsize=(8, 4))
        plt.plot(macd, label='MACD', color='blue')
        plt.plot(signal_line, label='Signal Line', color='orange')
        plt.title(f'{stock} - MACD')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        macd_img = io.BytesIO()
        plt.savefig(macd_img, format='png')
        plt.close()
        macd_img.seek(0)
        macd_base64 = base64.b64encode(macd_img.read()).decode()

        # --- Bollinger Bands ---
        ma, upper_band, lower_band = calculate_bollinger_bands(data['Close'])
        plt.figure(figsize=(8, 4))
        plt.plot(data['Close'], label='Close Price', color='black')
        plt.plot(ma, label='Middle Band', color='blue')
        plt.plot(upper_band, label='Upper Band', color='green')
        plt.plot(lower_band, label='Lower Band', color='red')
        plt.title(f'{stock} - Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        bb_img = io.BytesIO()
        plt.savefig(bb_img, format='png')
        plt.close()
        bb_img.seek(0)
        bb_base64 = base64.b64encode(bb_img.read()).decode()

        # --- HTML Template ---
        options_html = "".join(
            [f'<option value="{stk}" {"selected" if stk == stock else ""}>{stk}</option>' for stk in nifty_500]
        )
        interval_options = {"1d": "Daily", "1wk": "Weekly"}
        interval_html = "".join(
            [f'<option value="{ival}" {"selected" if ival == interval else ""}>{label}</option>'
             for ival, label in interval_options.items()]
        )
        forecast_html = "<ul style='list-style: none; padding: 0;'>"
        for key, val in result.items():
            forecast_html += f"<li><strong>{key}:</strong> {val}</li>"
        forecast_html += "</ul>"

        price_html = f"<p><strong>Current Market Price:</strong> ₹{round(current_price, 2) if isinstance(current_price, (float, int)) else current_price}</p>"

        html = f"""
        <html>
            <head>
                <title>Stock Forecast</title>
                <style>
                    body {{
                        background-color: #1E3A8A;
                        font-family: Arial, sans-serif;
                        color: #f1f1f1;
                        text-align: center;
                    }}
                    .container {{
                        margin: 40px auto;
                        padding: 20px;
                        max-width: 800px;
                        background-color: #2563EB;
                        border-radius: 10px;
                    }}
                    select, input[type=submit] {{
                        padding: 5px;
                        font-size: 14px;
                        margin: 10px;
                        border-radius: 5px;
                    }}
                    img {{
                        margin-top: 20px;
                        max-width: 100%;
                        border: 2px solid #ddd;
                        border-radius: 10px;
                    }}
                    ul li {{
                        font-size: 18px;
                        margin: 5px 0;
                    }}
                    p {{
                        font-size: 20px;
                        font-weight: bold;
                        color: #fffb00;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{stock} Forecast & Trend</h1>
                    {price_html}
                    <form method="get">
                        <label>Choose stock: </label>
                        <select name="stock">{options_html}</select>
                        <label>Interval: </label>
                        <select name="interval">{interval_html}</select>
                        <input type="submit" value="Update">
                    </form>
                    <h2>Forecast (Next Day OHLC)</h2>
                    {forecast_html}
                    <img src="data:image/png;base64,{bar_base64}" />
                    <h2>1-Year Price Trend</h2>
                    <img src="data:image/png;base64,{line_base64}" />
                    <h2>RSI (14-period)</h2>
                    <img src="data:image/png;base64,{rsi_base64}" />
                    <h2>MACD</h2>
                    <img src="data:image/png;base64,{macd_base64}" />
                    <h2>Bollinger Bands</h2>
                    <img src="data:image/png;base64,{bb_base64}" />
                </div>
            </body>
        </html>
        """
        return html

    except Exception as e:
        return f"<h2 style='color:red;'>Error: {str(e)}</h2>"

# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(debug=True)
