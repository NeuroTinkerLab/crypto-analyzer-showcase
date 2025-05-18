# AI-Generated Crypto Market Analysis Tool (Showcase/Archive)

This repository contains the Python source code for a comprehensive cryptocurrency market analysis tool. **The majority of this codebase was generated using Artificial Intelligence tools and is presented here primarily for showcase, archival, and educational purposes.**

**🚨 IMPORTANT DISCLAIMER 🚨**
*   **This code is UNVERIFIED, UNTESTED, and NOT INTENDED FOR PRODUCTION USE or making real financial decisions.**
*   It is provided "AS IS" without any warranties. Use at your own risk.
*   It likely contains bugs, inefficiencies, and may require significant modifications to be fully functional or reliable.
*   **DO NOT USE FOR LIVE TRADING.** Always do your own thorough research (DYOR).

## Project Goal

The tool is designed to:
1.  **Fetch Data:** Gather extensive market data from Binance (Spot/Futures) and Deribit (options/volatility).
2.  **Perform Analysis:** Conduct a wide array of analyses, including:
    *   Basic and advanced statistical analysis.
    *   Technical indicator calculations (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Volume Profile, etc.).
    *   Candlestick pattern detection.
    *   Fibonacci retracements, extensions, and time zones.
    *   Multi-Timeframe (MTF) context.
    *   Basic Machine Learning (Random Forest) for price direction prediction.
3.  **Generate Output:** Produce a structured JSON report encompassing all collected data and performed analyses. This JSON output is intended to be suitable for ingestion and interpretation by Large Language Models (LLMs) to potentially derive trading insights, market summaries, or further AI-driven analysis.

## Features (As per AI-Generated Code)

*   Data collection from Binance (OHLCV, Open Interest, Funding Rates, etc.) and Deribit (options, futures, volatility).
*   Calculation of numerous technical indicators.
*   Advanced statistical tests (stationarity, GARCH, cycle analysis, temporal bias).
*   Candlestick pattern recognition with quality assessment.
*   Fibonacci levels and time zone projections.
*   Multi-Timeframe (MTF) analysis.
*   Volume Profile analysis (current, daily, weekly).
*   A basic ML price direction predictor.
*   Output of all findings in a detailed JSON (and TXT) format.

## Prerequisites

*   Python 3.8+
*   Git (for cloning, if not uploading directly)

## Setup & (Potential) Usage

1.  **Clone the repository (optional if you downloaded a ZIP):**
    ```bash
    git clone https://github.com/NeuroTinkerLab/crypto-analyzer-showcase.git
    cd NOME_REPOSITORY
    ```
    
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Key Configuration (Optional):**
    The project is designed to fetch public data from Binance and Deribit without requiring API keys for many core functionalities.
    If you wish to test or use endpoints that *do* require authentication (e.g., for accessing private account data, though this script primarily focuses on public market data):
    *   The script `binance_data_fetcher.py` attempts to load `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` from your system's environment variables first.
    *   Alternatively, placeholder values for these keys are present in `config.py`. You would need to modify `config.py` **locally** (and ensure you do not commit these changes) with your actual keys if not using environment variables.
    *   **IMPORTANT: NEVER commit your actual API keys to GitHub. Ensure `config.py` in the repository only contains placeholder values like "YOUR_API_KEY_PLACEHOLDER".**

5.  **Running the Main Script (`main.py`):**
    The script can be run interactively or with command-line arguments.
    *   Interactive mode:
        ```bash
        python main.py
        ```
    *   Specific analysis via CLI:
        ```bash
        python main.py --pair BTC/USDT --timeframe 1h
        ```
    *   Update historical statistics for a pair/timeframe (requires prior data download):
        ```bash
        python main.py --pair ETH/USDT --timeframe 4h --update-historical-stats
        ```
    *   For all options:
        ```bash
        python main.py --help
        ```

    Reports are typically saved in `analysis_reports/`, historical statistics in `historical_stats/`, and OHLCV data is cached in `cache_analysis/`. These output directories should ideally be gitignored (they are included in the provided `.gitignore` file).

## Output for LLM and Processing Instructions

The primary output of this tool is a comprehensive JSON file generated for each selected symbol and timeframe. This JSON file contains all the fetched market data, calculated technical indicators, statistical measures, detected candlestick patterns, Fibonacci levels, multi-timeframe context, and other analytical insights.

The structure of the JSON is designed to be parsed and interpreted by a Large Language Model (LLM) to:
*   Summarize current market conditions.
*   Identify potential bullish or bearish scenarios.
*   Pinpoint key support, resistance, and target levels.
*   Evaluate the confluence of various technical signals across different timeframes.
*   Generate a concise trading outlook or potential trade ideas.

**Detailed System Prompt & LLM Operational Guidelines:**

For an in-depth example of the system prompt and detailed operational guidelines that can be used to instruct an LLM to process these JSON files and act as an "Algorithmic Multi-Timeframe Financial Analyst for Cryptocurrencies," please refer to the file:
➡️ **[LLM_SYSTEM_PROMPT_INSTRUCTIONS.md](LLM_SYSTEM_PROMPT_INSTRUCTIONS.md)** *(You will need to create this file and add your detailed LLM instructions to it).*

This linked file (once you create it) should contain the full, detailed prompt structure that outlines the expected role, competencies, guiding principles, operational workflow, and risk management considerations for the LLM when analyzing the generated JSON data. Users are encouraged to adapt and refine these instructions based on the specific LLM they are using and their analytical goals.

## Project Structure Overview

*   `main.py`: Main application entry point.
*   `config.py`: Central configuration file.
*   `data_collector.py`: Fetches historical OHLCV data (via ccxt).
*   `binance_data_fetcher.py`: Fetches specific data from Binance API.
*   `deribit_collector.py`: Fetches data from Deribit API.
*   `statistical_analyzer.py` & `statistical_analyzer_advanced.py`: Core statistical analysis modules.
*   `statistical_analyzer_helpers.py`: Utility functions for statistical analysis.
*   `statistical_analyzer_patterns.py`: Candlestick pattern detection.
*   `technical_analyzer.py`: Technical indicator calculations.
*   `fibonacci_calculator.py`: Fibonacci level and time zone calculations.
*   `timeframe_utils.py`: Utilities for Multi-Timeframe analysis.
*   `trading_advisor.py`: Orchestrates analyses and generates the final report.
*   `ml_predictor.py`: Basic ML model for price prediction.
*   `utils.py`: General utility functions.
*   `cache_analysis/`: (Gitignored) Cache for OHLCV data.
*   `historical_stats/`: (Gitignored) Cache for historical statistics.
*   `analysis_reports/`: (Gitignored) Output for analysis reports.
*   `requirements.txt`: Python dependencies.
*   `.gitignore`: Specifies intentionally untracked files by Git.
*   `LICENSE`: Project license (e.g., MIT License).

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.
*(Ensure you have a LICENSE file with the MIT License text, or your chosen license)*