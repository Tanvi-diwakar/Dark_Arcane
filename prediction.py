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
plt.savefig("plot.png")
plt.close()