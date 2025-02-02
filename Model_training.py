pip install numpy pandas yfinance tensorflow scikit-learn matplotlib

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].values.reshape(-1, 1)

# Example: Fetch Apple stock data
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'
stock_prices = fetch_stock_data(ticker, start_date, end_date)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_stock_prices = scaler.fit_transform(stock_prices)

# Create sequences for training
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_stock_prices, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict on test data
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Simulate a trading strategy
initial_investment = 10000  # $10,000
shares = 0
cash = initial_investment

for i in range(len(predicted_prices) - 1):
    current_price = y_test_actual[i]
    next_predicted_price = predicted_prices[i + 1]
    
    if next_predicted_price > current_price and cash >= current_price:
        # Buy
        shares += cash // current_price[0] # Access the first element of current_price
        cash -= (cash // current_price[0]) * current_price[0] # Access the first element of current_price
    elif next_predicted_price < current_price and shares > 0:
        # Sell
        cash += shares * current_price[0] # Access the first element of current_price
        shares = 0

# Final portfolio value
final_portfolio_value = cash + shares * y_test_actual[-1][0] # Access the first element of y_test_actual[-1]
print(f'Initial Investment: ${initial_investment}')
print(f'Final Portfolio Value: ${final_portfolio_value}')
# Extract the value from the NumPy array before formatting
print(f'Return: {((final_portfolio_value - initial_investment) / initial_investment) * 100:.2f}%')

# Simulate a trading strategy
initial_investment = 10000  # $10,000
shares = 0
cash = initial_investment

for i in range(len(predicted_prices) - 1):
    current_price = y_test_actual[i]
    next_predicted_price = predicted_prices[i + 1]
    
    if next_predicted_price > current_price and cash >= current_price:
        # Buy
        shares += cash // current_price[0] # Access the first element of current_price
        cash -= (cash // current_price[0]) * current_price[0] # Access the first element of current_price
    elif next_predicted_price < current_price and shares > 0:
        # Sell
        cash += shares * current_price[0] # Access the first element of current_price
        shares = 0

# Final portfolio value
final_portfolio_value = cash + shares * y_test_actual[-1][0] # Access the first element of y_test_actual[-1]
print(f'Initial Investment: ${initial_investment}')
print(f'Final Portfolio Value: ${final_portfolio_value}')
# Extract the value from the NumPy array before formatting
print(f'Return: {((final_portfolio_value - initial_investment) / initial_investment) * 100:.2f}%')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Prepare data
def prepare_data(stock_data):
    # Use 'Open', 'High', 'Low', 'Volume' as features and 'Close' as the target
    features = stock_data[['Open', 'High', 'Low', 'Volume']]
    target = stock_data['Close']
    return features, target

# Step 3: Train and test the model
def train_and_test_model(features, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    
    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    
    # Plot the results
    plt.figure(figsize=(14,5))
    plt.plot(y_test.index, y_test, color='blue', label='Actual Stock Price')
    plt.plot(y_test.index, y_pred, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    return y_test, y_pred

# Step 4: Simulate a trading strategy
def simulate_trading(y_test, y_pred, initial_investment=10000):
    shares = 0
    cash = initial_investment
    
    for i in range(1, len(y_pred)):
        current_price = y_test.iloc[i-1]
        next_predicted_price = y_pred[i]
        
        if next_predicted_price > current_price and cash >= current_price:
            # Buy
            shares += cash // current_price
            cash -= (cash // current_price) * current_price
        elif next_predicted_price < current_price and shares > 0:
            # Sell
            cash += shares * current_price
            shares = 0
    
    # Final portfolio value
    final_portfolio_value = cash + shares * y_test.iloc[-1]
    print(f'Initial Investment: ${initial_investment}')
    print(f'Final Portfolio Value: ${final_portfolio_value}')
    print(f'Return: {((final_portfolio_value - initial_investment) / initial_investment) * 100:.2f}%')

# Main program
if __name__ == "__main__":
    # Fetch data
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    features, target = prepare_data(stock_data)
    
    # Train and test the model
    y_test, y_pred = train_and_test_model(features, target)
    
    # Simulate trading
    simulate_trading(y_test, y_pred)

    from ipaddress import IPv4Network, IPv4Address

# Define an IP network
network = IPv4Network("192.168.1.0/24")

# Print network details
print(f"Network Address: {network.network_address}")
print(f"Broadcast Address: {network.broadcast_address}")
print(f"Subnet Mask: {network.netmask}")
print(f"Number of Hosts: {network.num_addresses}")

# Check if an IP is in the network
ip = IPv4Address("192.168.1.10")
print(f"Is {ip} in {network}? {ip in network}")

from ipaddress import IPv4Address, IPv4Network

# Define a routing table
routing_table = {
    "192.168.1.0/24": "Router1",
    "10.0.0.0/8": "Router2",
    "172.16.0.0/16": "Router3",
    "0.0.0.0/0": "DefaultGateway"  # Default route
}

# Function to find the next hop for a given IP
def find_next_hop(ip, routing_table):
    ip = IPv4Address(ip)
    for network, next_hop in routing_table.items():
        if ip in IPv4Network(network):
            return next_hop
    return "No route found"

# Test the routing function
destination_ip = "172.16.0.1"
next_hop = find_next_hop(destination_ip, routing_table)
print(f"Next hop for {destination_ip}: {next_hop}")

# Define routers and their interfaces
routers = {
    "Router1": {"192.168.1.1": "Local", "10.0.0.1": "Router2"},
    "Router2": {"10.0.0.1": "Router1", "172.16.0.1": "Router3"},
    "Router3": {"172.16.0.1": "Router2", "192.168.2.1": "Local"}
}

# Function to forward a packet
def forward_packet(source, destination, routers):
    current_router = source
    path = [current_router]

    while current_router != destination:
        if current_router not in routers:
            return "Destination unreachable"
        
        # Find the next hop
        next_hop = None
        for interface, neighbor in routers[current_router].items():
            if neighbor == destination:
                next_hop = destination
                break
            elif neighbor != "Local":
                next_hop = neighbor
        
        if not next_hop:
            return "Destination unreachable"
        
        current_router = next_hop
        path.append(current_router)
    
    return path

# Test packet forwarding
source = "Router1"
destination = "Router3"
path = forward_packet(source, destination, routers)
print(f"Path from {source} to {destination}: {path}")

    "Router1": {"192.168.1.1": "Local", "10.0.0.1": "Router2"},

# Define a routing table as a dictionary
routing_table = {
    "192.168.1.0/24": "Router1",
    "10.0.0.0/8": "Router2",
    "0.0.0.0/0": "DefaultGateway"  # Default route
}

# Function to find the next hop for a given IP
def find_next_hop(ip, routing_table):
    from ipaddress import IPv4Address, IPv4Network

    ip = IPv4Address(ip)
    for network, next_hop in routing_table.items():
        if ip in IPv4Network(network):
            return next_hop
    return "No route found"

# Test the routing function
destination_ip = "192.168.1.10"
next_hop = find_next_hop(destination_ip, routing_table)
print(f"Next hop for {destination_ip}: {next_hop}")

import os

# Function to simulate ICMP ping
def ping(host):
    response = os.system(f"ping -c 1 {host} > /dev/null 2>&1")
    if response == 0:
        print(f"{host} is reachable")
    else:
        print(f"{host} is unreachable")

# Test ping
ping("192.168.1.1")  # Replace with a valid IP address

# Define an ARP table
arp_table = {
    "192.168.1.10": "00:1A:2B:3C:4D:5E",
    "192.168.1.11": "00:1A:2B:3C:4D:5F",
}

# Function to resolve MAC address from IP
def resolve_mac(ip):
    return arp_table.get(ip, "Unknown MAC address")

# Test ARP resolution
ip = "192.168.1.10"
mac = resolve_mac(ip)
print(f"MAC address for {ip}: {mac}")

class Router:
    def __init__(self, name, interfaces):
        self.name = name
        self.interfaces = interfaces  # {ip: network}

    def forward(self, packet):
        destination_ip = packet["destination_ip"]
        for interface_ip, network in self.interfaces.items():
            if IPv4Address(destination_ip) in IPv4Network(network):
                print(f"{self.name} forwarding packet to {destination_ip} via {interface_ip}")
                return
        print(f"{self.name} has no route to {destination_ip}")

# Define routers
router1 = Router("Router1", {"192.168.1.1": "192.168.1.0/24", "10.0.0.1": "10.0.0.0/8"})
router2 = Router("Router2", {"10.0.0.1": "10.0.0.0/8", "172.16.0.1": "172.16.0.0/16"})

# Simulate packet forwarding
packet = {"source_ip": "192.168.1.10", "destination_ip": "172.16.0.10"}
router1.forward(packet)
router2.forward(packet)

!python layer3.py

import matplotlib.pyplot as plt
import pandas as pd

# Sample data (replace with your actual data)
data = {'Date': ['2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'],
        'Predicted Price': [150.0, 155.0, 160.0, 165.0, 170.0]}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime objects

# Create the plot
plt.plot(df['Date'], df['Predicted Price'], marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.title('Predicted Price Trend')

# Customize appearance (optional)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# Show the plot
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()