from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import logging
import datetime
import os
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Dummy data generator functions
def generate_dummy_predictions(ticker):
    """Generate dummy prediction data for demonstration purposes"""
    today = datetime.datetime.now()
    
    # Generate dates for the next 7 days
    dates = [(today + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
    
    # Start with a reasonable stock price (random between $50-$200)
    base_price = np.random.randint(50, 200)
    
    # Generate slightly varying prices with a trend
    trend = np.random.choice([-1, 1]) * 0.005  # Random trend direction
    prices = [base_price]
    
    for i in range(6):
        # Add some random variation plus the trend
        new_price = prices[-1] * (1 + trend + np.random.normal(0, 0.01))
        prices.append(round(new_price, 2))
    
    # Create historical prices (last 30 days)
    historical_dates = [(today - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    
    # Generate historical prices with some randomness
    historical_prices = [base_price]
    for i in range(29):
        historical_prices.append(round(historical_prices[-1] * (1 + np.random.normal(0, 0.01)), 2))
    historical_prices.reverse()  # Most recent dates last
    
    # Format predictions
    predictions = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        prev_price = prices[i-1] if i > 0 else historical_prices[-1]
        change = round((price - prev_price) / prev_price * 100, 2)
        
        predictions.append({
            'date': date,
            'price': f"{price:.2f}",
            'change': change,
            'confidence': round(95 - i * 5, 2)  # Lower confidence for later predictions
        })
    
    return predictions, dates, {
        'historical_dates': historical_dates,
        'historical_prices': historical_prices,
        'current_price': historical_prices[-1],
        'price_change': round((historical_prices[-1] - historical_prices[-2]) / historical_prices[-2] * 100, 2),
        'company_name': f"{ticker} Inc."
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    
    if request.method == 'POST':
        ticker = request.form['ticker']
        
        if not ticker:
            error = "Please enter a stock ticker symbol"
        else:
            # Generate dummy predictions
            predictions, dates, historical_data = generate_dummy_predictions(ticker)
            
            # Render the results template
            return render_template('result.html',
                ticker=ticker,
                company_name=historical_data['company_name'],
                current_price=historical_data['current_price'],
                price_change=historical_data['price_change'],
                predictions=predictions,
                historical_prices=historical_data['historical_prices'],
                historical_dates=historical_data['historical_dates'],
                accuracy=92.5,
                training_days=365
            )
    
    return render_template('index.html', error=error)

@app.route('/download_csv/<ticker>')
def download_csv(ticker):
    # Generate dummy predictions
    predictions, _, _ = generate_dummy_predictions(ticker)
    
    # Create DataFrame
    df = pd.DataFrame(predictions)
    
    # Create a string buffer
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"{ticker}_predictions.csv"
    )

@app.route('/download_pdf/<ticker>')
def download_pdf(ticker):
    # In a real app, you'd generate a PDF here
    # For demo purposes, just return a text file
    return f"This would be a PDF report for {ticker}", 200

if __name__ == '__main__':
    app.run(debug=True)