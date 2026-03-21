import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from scipy.stats import norm
import numpy as np
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import yfinance as yf

# --- CONFIGURATION ---
SHEET_NAME      = 'Live Option Chain'
JSON_KEYFILE    = 'credentials.json'
REFRESH_SECONDS = 60
# Auto-set to True when running on GitHub Actions (set via env var)
import os
HEADLESS_MODE   = os.environ.get('HEADLESS_MODE', 'false').lower() == 'true'

# Symbol config: name → sheet tab name
SYMBOLS = {
    'NIFTY':     'NiftyData',
    'BANKNIFTY': 'BankniftyData',
}

# Nifty 100 constituents sheet tab
NIFTY100_TAB = 'Nifty100Data'
# NSE URL for Nifty 100 index constituents
NIFTY100_URL = 'https://www.nseindia.com/products-services/indices-nifty100-index'
# NSE API endpoint for Nifty 100 constituents (faster than Selenium)
NIFTY100_API = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100'
# Columns to extract from NSE API response
NIFTY100_COLS = ['symbol', 'previousClose', 'open', 'lastPrice']
NIFTY100_HEADERS = ['Symbol', 'Prev. Close', 'Open', 'LTP']

# Strike step per symbol (used for pre-open rounding)
STRIKE_STEP = {
    'NIFTY':     50,
    'BANKNIFTY': 100,
}

# yfinance ticker symbols for open price fetch
YFINANCE_TICKER = {
    'NIFTY':     '^NSEI',
    'BANKNIFTY': '^NSEBANK',
}

# --- SCHEDULE CONFIGURATION (IST) ---
BOT_START_TIME    = (9, 15)   # Bot starts recording at market open
MARKET_OPEN_TIME  = (9, 15)   # Full option chain recording begins
MARKET_CLOSE_TIME = (15, 30)  # Bot stops recording and exits

# Per-symbol open price state — captured once at startup, never overwritten
_open_price = {
    'NIFTY':     {'price': None, 'strike': None, 'time': None, 'source': None},
    'BANKNIFTY': {'price': None, 'strike': None, 'time': None, 'source': None},
}


# ==============================================================
#  TIME HELPERS
# ==============================================================

def get_ist_now():
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    )

def time_to_mins(h, m):
    return h * 60 + m

def ist_mins():
    t = get_ist_now()
    return t.hour * 60 + t.minute

def is_weekend():
    return get_ist_now().weekday() >= 5

def is_before_start():
    return ist_mins() < time_to_mins(*BOT_START_TIME)

def is_market_open():
    mins = ist_mins()
    return time_to_mins(*MARKET_OPEN_TIME) <= mins < time_to_mins(*MARKET_CLOSE_TIME)

def is_after_close():
    return ist_mins() >= time_to_mins(*MARKET_CLOSE_TIME)

def seconds_until(h, m):
    t      = get_ist_now()
    target = t.replace(hour=h, minute=m, second=0, microsecond=0)
    if target < t:
        target += datetime.timedelta(days=1)
    return int((target - t).total_seconds())

def fmt_mins(mins):
    return f"{mins // 60:02d}:{mins % 60:02d}"


# ==============================================================
#  OPEN PRICE LOGIC
# ==============================================================

def round_to_nearest_strike(price, step):
    return int(round(price / step) * step)

def _store_open_price(symbol, open_price, source):
    """Helper: store open price, strike, time and source into _open_price dict."""
    step   = STRIKE_STEP.get(symbol, 50)
    strike = round_to_nearest_strike(open_price, step)
    t_str  = get_ist_now().strftime('%H:%M:%S')
    _open_price[symbol]['price']  = open_price
    _open_price[symbol]['strike'] = strike
    _open_price[symbol]['time']   = t_str
    _open_price[symbol]['source'] = source
    print(f"   -> [OPEN PRICE] [{symbol}] Source               : {source}")
    print(f"   -> [OPEN PRICE] [{symbol}] Open price           : {open_price:,.2f}")
    print(f"   -> [OPEN PRICE] [{symbol}] Nearest strike (+/-{step}): {strike}")
    print(f"   -> [OPEN PRICE] [{symbol}] Captured at          : {t_str}")

# --- NIFTY: scrape open price from NSE indices JSON API ---
def fetch_open_price_nse(symbol):
    """
    Fetch today's official open price for NIFTY from NSE's indices JSON API.
    Uses requests with NSE cookie headers — no Selenium needed.
    Returns open price as float, or 0 on failure.
    """
    import requests as req
    try:
        session = req.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/',
        }
        # Step 1: hit homepage to set cookies
        session.get('https://www.nseindia.com', headers=headers, timeout=10)

        # Step 2: fetch indices data — contains open price for all NSE indices
        url  = 'https://www.nseindia.com/api/allIndices'
        resp = session.get(url, headers=headers, timeout=10)
        data = resp.json()

        # Search for NIFTY 50 entry
        for item in data.get('data', []):
            index_name = item.get('index', '').upper()
            if index_name == 'NIFTY 50':
                open_val = float(item.get('open', 0))
                print(f"   -> [NSE API] [{symbol}] Official open price: {open_val:,.2f}")
                return open_val

        print(f"   -> [NSE API] [{symbol}] NIFTY 50 not found in response.")
        return 0

    except Exception as e:
        print(f"   -> [NSE API] [{symbol}] Error: {e}")
        return 0

# --- BANKNIFTY: fetch open price via yfinance ---
def fetch_open_price_yfinance(symbol):
    """
    Fetch today's official open price for BANKNIFTY using yfinance.
    Returns open price as float, or 0 on failure.
    """
    ticker_sym = YFINANCE_TICKER.get(symbol)
    if not ticker_sym:
        print(f"   -> [yfinance] No ticker mapped for {symbol}")
        return 0
    try:
        ticker = yf.Ticker(ticker_sym)
        data   = ticker.history(period='1d', interval='1m')
        if data.empty:
            print(f"   -> [yfinance] No data returned for {ticker_sym}")
            return 0
        open_price = float(data['Open'].iloc[0])
        print(f"   -> [yfinance] [{symbol}] Official open price: {open_price:,.2f}")
        return open_price
    except Exception as e:
        print(f"   -> [yfinance] [{symbol}] Error: {e}")
        return 0

def capture_open_price(symbol):
    """
    Fetch and store today's official open price.
    - NIFTY     → NSE indices API (nseindia.com)
    - BANKNIFTY → yfinance
    Falls back to first scraped spot if both fail.
    Called once at startup — never overwritten after that.
    """
    if _open_price[symbol]['price'] is not None:
        return

    if symbol == 'NIFTY':
        open_price = fetch_open_price_nse(symbol)
        source     = 'NSE API'
    else:
        open_price = fetch_open_price_yfinance(symbol)
        source     = 'yfinance'

    if open_price > 0:
        _store_open_price(symbol, open_price, source)
    else:
        print(f"   -> [OPEN PRICE] [{symbol}] Fetch failed — "
              f"will use first scraped spot as fallback.")

def capture_open_price_fallback(symbol, spot_price):
    """
    Fallback: if primary fetch failed, use first scraped spot price.
    Called inside fetch_live_data — only stores if not yet set.
    """
    if _open_price[symbol]['price'] is None and spot_price > 0:
        print(f"   -> [FALLBACK] [{symbol}] Using first scraped spot: {spot_price:,.2f}")
        _store_open_price(symbol, spot_price, 'fallback')


# ==============================================================
#  GOOGLE SHEETS
# ==============================================================

def connect_to_sheet(tab_name):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds  = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEYFILE, scope)
    client = gspread.authorize(creds)
    wb     = client.open(SHEET_NAME)
    try:
        sheet = wb.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        sheet = wb.add_worksheet(title=tab_name, rows=300, cols=30)
        print(f"   -> Created new sheet tab: {tab_name}")
    return sheet

def write_row1(sheet, symbol, spot_price):
    """
    Write Row 1 with label and value in separate cells:
    A1: Label   B1: Value   C1: Label   D1: Value  ... and so on
    """
    step    = STRIKE_STEP.get(symbol, 50)
    now_str = get_ist_now().strftime('%H:%M:%S')
    op      = _open_price[symbol]

    if op['price'] is not None:
        open_price_val  = op['price']   # numeric — no comma formatting
        open_strike_val = op['strike']  # integer
        open_time_val   = op['time']
    else:
        open_price_val  = "Not captured yet"
        open_strike_val = "—"
        open_time_val   = "—"

    # Each pair: label cell, value cell
    row1 = [
        "Symbol",          symbol,
        "Spot",            round(spot_price, 2),
        "Last Updated",    now_str,
        "Open Price",      open_price_val,
        f"Open Strike (nearest {step})", open_strike_val,
        "Open Time",       open_time_val,
        "Open Source",     op.get('source') or "—",
    ]
    sheet.update([row1], 'A1')
    print(f"   -> [{symbol}] Row 1 written ({len(row1)} cells, label+value pairs)")


# ==============================================================
#  BLACK-SCHOLES DELTA
# ==============================================================

def calculate_delta(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma == 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'CE':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


# ==============================================================
#  PARSE HELPER
# ==============================================================

def parse(val):
    v = str(val).strip()
    if v in ['-', '', 'nan', 'NaN', 'None', '-\n-']:
        return 0
    v = v.replace(',', '').replace('\n', ' ').split(' ')[0]
    try:
        return float(v)
    except:
        return 0


# ==============================================================
#  DROPDOWN SELECTOR
# ==============================================================

def select_symbol_from_dropdown(driver, symbol):
    print(f"   -> [{symbol}] Locating 'View Options Contracts for:' dropdown...")
    try:
        dropdown = None
        selectors_to_try = [
            (By.ID,           "equity_optionChain_select"),
            (By.ID,           "optionChainIndexSelect"),
            (By.ID,           "indexType"),
            (By.CSS_SELECTOR, "select[name='indexType']"),
            (By.CSS_SELECTOR, "select.option-chain-select"),
            (By.XPATH, "//label[contains(text(),'View Options Contracts')]"
                       "/following-sibling::select"),
            (By.XPATH, "//label[contains(text(),'View Options Contracts')]"
                       "/..//select"),
            (By.XPATH, "//select[contains(@class,'form-control')]"),
        ]
        for by, selector in selectors_to_try:
            try:
                el = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((by, selector))
                )
                dropdown = el
                print(f"   -> [{symbol}] Dropdown found via: {selector}")
                break
            except:
                continue

        if dropdown is None:
            print(f"   -> [{symbol}] Fallback: scanning all <select> elements...")
            for sel_el in driver.find_elements(By.TAG_NAME, "select"):
                try:
                    opts = [o.text.strip().upper()
                            for o in sel_el.find_elements(By.TAG_NAME, "option")]
                    if any(symbol.upper() in o for o in opts):
                        dropdown = sel_el
                        print(f"   -> [{symbol}] Matched dropdown. Options: {opts}")
                        break
                except:
                    continue

        if dropdown is None:
            print(f"   -> [{symbol}] [Error] Dropdown not found.")
            return False

        driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
        time.sleep(0.5)

        select_obj   = Select(dropdown)
        matched_text = None
        for option in select_obj.options:
            opt_text = option.text.strip().upper()
            opt_val  = (option.get_attribute("value") or "").strip().upper()
            if symbol.upper() in opt_text or symbol.upper() in opt_val:
                matched_text = option.text.strip()
                break

        if matched_text is None:
            print(f"   -> [{symbol}] [Error] Not in options: "
                  f"{[o.text.strip() for o in select_obj.options]}")
            return False

        select_obj.select_by_visible_text(matched_text)
        print(f"   -> [{symbol}] Selected: '{matched_text}'")
        time.sleep(3)

        try:
            WebDriverWait(driver, 30).until(
                lambda d: len(d.find_elements(
                    By.CSS_SELECTOR, "#optionChainTable-indices tbody tr"
                )) > 5
            )
            print(f"   -> [{symbol}] Table reloaded!")
        except:
            print(f"   -> [{symbol}] [Warning] Table reload timed out.")

        time.sleep(2)
        return True

    except Exception as e:
        print(f"   -> [{symbol}] [Error] Dropdown failed: {e}")
        return False


# ==============================================================
#  BROWSER — FETCH LIVE DATA
# ==============================================================

def fetch_live_data(symbol):
    print(f"   -> [{symbol}] Launching Chrome (Stealth Mode)...")
    opts = Options()
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    if HEADLESS_MODE:
        opts.add_argument("--headless=new")
    else:
        opts.add_argument("--window-position=-32000,-32000")
        opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    })

    try:
        print(f"   -> [{symbol}] Visiting NSE homepage to set cookies...")
        driver.get("https://www.nseindia.com")
        time.sleep(4)

        print(f"   -> [{symbol}] Loading option chain page...")
        driver.get("https://www.nseindia.com/option-chain")

        try:
            WebDriverWait(driver, 40).until(
                EC.presence_of_element_located((By.ID, "optionChainTable-indices"))
            )
            print(f"   -> [{symbol}] Table container detected!")
        except:
            print(f"   [{symbol}] [Error] Table load timeout.")
            return None, None

        try:
            WebDriverWait(driver, 40).until(
                lambda d: len(d.find_elements(
                    By.CSS_SELECTOR, "#optionChainTable-indices tbody tr"
                )) > 5
            )
            print(f"   -> [{symbol}] Initial rows detected!")
        except:
            print(f"   -> [{symbol}] [Warning] Rows slow, proceeding...")

        time.sleep(2)

        if symbol != 'NIFTY':
            success = select_symbol_from_dropdown(driver, symbol)
            if not success:
                print(f"   -> [{symbol}] [Warning] Dropdown switch failed.")

        try:
            spot_element = driver.find_element(By.ID, "equity_underlyingVal")
            spot_text = (
                spot_element.text
                .replace("Underlying Index: ", "")
                .replace("BANKNIFTY", "")
                .replace("NIFTY", "")
                .strip()
            )
            spot_price = float(spot_text.split(" ")[0].replace(",", ""))
            print(f"   -> [{symbol}] Spot Price: {spot_price:,.2f}")
        except:
            spot_price = 0
            print(f"   -> [{symbol}] WARNING: Could not extract spot price.")

        # Fallback: if yfinance failed at startup, use first scraped spot
        capture_open_price_fallback(symbol, spot_price)

        print(f"   -> [{symbol}] Extracting table data via JavaScript...")
        rows_data = driver.execute_script("""
            var table = document.getElementById('optionChainTable-indices');
            if (!table) return [];
            var rows = table.querySelectorAll('tbody tr');
            var result = [];
            rows.forEach(function(row) {
                var cells = row.querySelectorAll('td');
                var rowData = [];
                cells.forEach(function(cell) {
                    rowData.push(cell.innerText.trim());
                });
                if (rowData.length === 23) { result.push(rowData); }
            });
            return result;
        """)

        print(f"   -> [{symbol}] Total rows extracted: {len(rows_data)}")

        if not rows_data or len(rows_data) < 3:
            print(f"   [{symbol}] [Error] Not enough rows extracted.")
            return None, None

        return rows_data, spot_price

    except Exception as e:
        print(f"   [{symbol}] [Browser Error] {e}")
        return None, None

    finally:
        driver.quit()
        print(f"   -> [{symbol}] Browser closed.")


# ==============================================================
#  PROCESS & WRITE ONE SYMBOL
# ==============================================================

def process_symbol(symbol, tab_name):
    print(f"\n   ===== Processing: {symbol} → {tab_name} =====")

    rows_data, spot_price = fetch_live_data(symbol)

    if not rows_data:
        print(f"   -> [{symbol}] No data fetched. Skipping.")
        return

    try:
        clean_data = []
        T = 4.0 / 365.0

        for row in rows_data:
            try:
                c_oi      = parse(row[1]);  c_chng_oi = parse(row[2])
                c_vol     = parse(row[3]);  c_iv      = parse(row[4])
                c_ltp     = parse(row[5]);  c_chng    = parse(row[6])
                c_bid_qty = parse(row[7]);  c_bid     = parse(row[8])
                c_ask     = parse(row[9]);  c_ask_qty = parse(row[10])
                strike    = parse(row[11])
                p_bid_qty = parse(row[12]); p_bid     = parse(row[13])
                p_ask     = parse(row[14]); p_ask_qty = parse(row[15])
                p_chng    = parse(row[16]); p_ltp     = parse(row[17])
                p_iv      = parse(row[18]); p_vol     = parse(row[19])
                p_chng_oi = parse(row[20]); p_oi      = parse(row[21])

                if strike <= 0:
                    continue

                c_delta = calculate_delta(spot_price, strike, T, 0.10, c_iv / 100, 'CE')
                p_delta = calculate_delta(spot_price, strike, T, 0.10, p_iv / 100, 'PE')

                clean_data.append([
                    c_oi, c_chng_oi, c_vol, c_iv, round(c_delta, 2),
                    c_ltp, c_chng, c_bid_qty, c_bid, c_ask, c_ask_qty,
                    strike,
                    p_bid_qty, p_bid, p_ask, p_ask_qty, p_chng,
                    p_ltp, p_iv, round(p_delta, 2), p_vol, p_chng_oi, p_oi
                ])
            except Exception as row_err:
                print(f"   -> [{symbol}] Skipping row: {row_err}")
                continue

        print(f"   -> [{symbol}] Clean rows: {len(clean_data)}")

        if not clean_data:
            print(f"   -> [{symbol}] ERROR: No valid rows.")
            return

        headers = [
            'Call OI', 'Call Chng OI', 'Call Vol', 'Call IV', 'Call Delta',
            'Call LTP', 'Call Chng', 'Call Bid Qty', 'Call Bid', 'Call Ask', 'Call Ask Qty',
            'Strike Price',
            'Put Bid Qty', 'Put Bid', 'Put Ask', 'Put Ask Qty', 'Put Chng',
            'Put LTP', 'Put IV', 'Put Delta', 'Put Vol', 'Put Chng OI', 'Put OI'
        ]

        final_df = pd.DataFrame(clean_data, columns=headers)
        sheet    = connect_to_sheet(tab_name)
        sheet.clear()
        write_row1(sheet, symbol, spot_price)
        sheet.update([final_df.columns.tolist()], 'A2')
        sheet.update(final_df.values.tolist(), 'A3')

        print(f"   -> [{symbol}] SUCCESS! {len(final_df)} rows written to '{tab_name}'.")

    except Exception as e:
        print(f"   -> [{symbol}] Processing Error: {e}")
        import traceback
        traceback.print_exc()



# ==============================================================
#  NIFTY 100 SCRAPER
# ==============================================================

def fetch_nifty100_data():
    """
    Fetch Nifty 100 constituents data from NSE via:
      Home → Market Data → Indices → Nifty 100
    Uses NSE JSON API (same approach as open price fetch — lightweight, no Selenium).
    Returns a list of [Symbol, Prev. Close, Open, LTP] rows, or None on failure.
    """
    import requests as req
    print(f"   -> [Nifty100] Fetching Nifty 100 constituents from NSE API...")
    try:
        session = req.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/',
        }

        # Step 1: Hit homepage to get NSE session cookies
        print(f"   -> [Nifty100] Setting NSE session cookies...")
        session.get('https://www.nseindia.com', headers=headers, timeout=15)

        # Step 2: Hit market data / indices page to warm up referer cookies
        session.get('https://www.nseindia.com/market-data/live-equity-market',
                    headers=headers, timeout=10)

        # Step 3: Call NSE equity-stockIndices API for NIFTY 100
        print(f"   -> [Nifty100] Calling NSE indices API...")
        resp = session.get(NIFTY100_API, headers=headers, timeout=15)

        if resp.status_code != 200:
            print(f"   -> [Nifty100] [Error] API returned status {resp.status_code}")
            return None

        data  = resp.json()
        items = data.get('data', [])

        if not items:
            print(f"   -> [Nifty100] [Error] No data in API response.")
            return None

        rows = []
        for item in items:
            try:
                symbol     = str(item.get('symbol',       '')).strip()
                prev_close = item.get('previousClose', 0)
                open_val   = item.get('open',          0)
                ltp        = item.get('lastPrice',     0)

                # Skip the index summary row (NSE includes NIFTY 100 itself as first row)
                if symbol in ('NIFTY 100', 'NIFTY100', ''):
                    continue

                rows.append([
                    symbol,
                    round(float(prev_close), 2),
                    round(float(open_val),   2),
                    round(float(ltp),        2),
                ])
            except Exception as row_err:
                print(f"   -> [Nifty100] Skipping row: {row_err}")
                continue

        print(f"   -> [Nifty100] Total constituents fetched: {len(rows)}")
        return rows

    except Exception as e:
        print(f"   -> [Nifty100] [Error] {e}")
        return None


def process_nifty100():
    """
    Fetch Nifty 100 data and write to the Nifty100 sheet tab.
    Writes:
      Row 1 : Info header (last updated time)
      Row 2 : Column headers — Symbol | Prev. Close | Open | LTP
      Row 3+ : Data rows
    """
    print(f"\n   ===== Processing: Nifty 100 → {NIFTY100_TAB} =====")

    rows = fetch_nifty100_data()

    if not rows:
        print(f"   -> [Nifty100] No data fetched. Skipping this cycle.")
        return

    try:
        sheet    = connect_to_sheet(NIFTY100_TAB)
        now_str  = get_ist_now().strftime('%H:%M:%S')

        sheet.clear()

        # Row 1: info header (label + value pairs)
        sheet.update([
            ['Index', 'NIFTY 100',
             'Constituents', len(rows),
             'Last Updated', now_str]
        ], 'A1')

        # Row 2: column headers
        sheet.update([NIFTY100_HEADERS], 'A2')

        # Row 3 onwards: data
        sheet.update(rows, 'A3')

        print(f"   -> [Nifty100] SUCCESS! {len(rows)} rows written to '{NIFTY100_TAB}'.")

    except Exception as e:
        print(f"   -> [Nifty100] Processing Error: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================
#  CYCLE RUNNERS
# ==============================================================

def run_cycle():
    """Run one full update cycle for all symbols + Nifty 100."""
    now_str = get_ist_now().strftime('%H:%M:%S')
    print(f"\n[{now_str}] ===== Update Cycle =====")
    for symbol, tab_name in SYMBOLS.items():
        process_symbol(symbol, tab_name)
    process_nifty100()
    print(f"[{get_ist_now().strftime('%H:%M:%S')}] Cycle complete.")



def run_one_time_cycle():
    """
    Run a single data fetch cycle for all symbols + Nifty 100 and exit.
    Used when bot is started after market close or on a weekend.
    """
    print(f"\n  Fetching one-time snapshot for all symbols...\n")
    for symbol, tab_name in SYMBOLS.items():
        process_symbol(symbol, tab_name)
    process_nifty100()
    print(f"\n[{get_ist_now().strftime('%H:%M:%S')}] One-time snapshot complete.")


# ==============================================================
#  ENTRY POINT — SMART SCHEDULER
# ==============================================================

if __name__ == "__main__":

    print("=" * 62)
    print("  NSE Option Chain Bot — Smart Scheduler")
    print("=" * 62)
    for sym, tab in SYMBOLS.items():
        print(f"  {sym:12s} → {tab:10s} | Strike step: {STRIKE_STEP[sym]}")
    print(f"  Recording    : {fmt_mins(time_to_mins(*MARKET_OPEN_TIME))}–"
          f"{fmt_mins(time_to_mins(*MARKET_CLOSE_TIME))} IST")
    print(f"  Open price   : Captured on first cycle (9:15 AM)")
    print(f"  Refresh      : {REFRESH_SECONDS}s")
    print(f"  Auto-stop    : {fmt_mins(time_to_mins(*MARKET_CLOSE_TIME))} IST")
    print("=" * 62)

    try:

        # --- FETCH OPEN PRICES VIA YFINANCE AT STARTUP ---
        print(f"\n  Fetching official open prices via yfinance...")
        for symbol in SYMBOLS:
            capture_open_price(symbol)

        # --- WEEKEND: run one snapshot then exit ---
        if is_weekend():
            day_name = get_ist_now().strftime('%A')
            print(f"\n  Today is {day_name} (weekend). NSE is closed.")
            print(f"  Running one-time data snapshot instead...\n")
            run_one_time_cycle()
            print("\n  Done. Exiting.\n")
            exit(0)

        # --- AFTER MARKET CLOSE: run one snapshot then exit ---
        if is_after_close():
            print(f"\n  Market already closed for today (past 3:30 PM IST).")
            print(f"  Running one-time data snapshot instead...\n")
            run_one_time_cycle()
            print("\n  Done. Exiting.\n")
            exit(0)

        # --- BEFORE 9:15 AM: wait with live countdown ---
        if is_before_start():
            wait_secs = seconds_until(*BOT_START_TIME)
            wake_time = fmt_mins(time_to_mins(*BOT_START_TIME))
            print(f"\n  Current IST time : {get_ist_now().strftime('%H:%M:%S')}")
            print(f"  Waiting until    : {wake_time} IST  ({wait_secs}s)")
            print(f"  Bot will auto-start at {wake_time} AM. "
                  f"Press Ctrl+C to cancel.\n")
            while is_before_start():
                remaining  = seconds_until(*BOT_START_TIME)
                mins_left  = remaining // 60
                secs_left  = remaining % 60
                print(f"  [{get_ist_now().strftime('%H:%M:%S')}] "
                      f"Starting in {mins_left}m {secs_left}s ...", end='\r')
                time.sleep(30)
            print(f"\n\n  [{get_ist_now().strftime('%H:%M:%S')}] "
                  f"9:15 AM reached — Bot starting now!")

        # --- MAIN LOOP ---
        print(f"\n  [{get_ist_now().strftime('%H:%M:%S')}] Bot is RUNNING.\n")

        while True:

            # AUTO STOP at 3:30 PM
            if is_after_close():
                print("\n" + "=" * 62)
                print(f"  [{get_ist_now().strftime('%H:%M:%S')}] "
                      f"3:30 PM IST reached — Market closed.")
                print(f"  Bot has finished recording for today. Shutting down.")
                print("=" * 62)
                break

            # MARKET HOURS 9:15 AM–3:30 PM
            elif is_market_open():
                run_cycle()
                remaining_to_close = seconds_until(*MARKET_CLOSE_TIME)
                sleep_time = min(REFRESH_SECONDS, remaining_to_close)
                if sleep_time > 0:
                    print(f"   -> Next cycle in {sleep_time}s "
                          f"(market closes in "
                          f"{remaining_to_close // 60}m "
                          f"{remaining_to_close % 60}s).\n")
                    time.sleep(sleep_time)

            else:
                print(f"  [{get_ist_now().strftime('%H:%M:%S')}] "
                      f"Waiting for market open at 09:15...", end='\r')
                time.sleep(10)

    except KeyboardInterrupt:
        print(f"\n\n  [{get_ist_now().strftime('%H:%M:%S')}] "
              f"Bot manually stopped by user.")

    print("\n  Session ended. Goodbye!\n")
