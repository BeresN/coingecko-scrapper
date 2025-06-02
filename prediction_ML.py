import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEQUENCE_LENGTH = 30
filepath = "ohlcv.csv"

# --- Data Preprocessing ---
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['token_id', 'timestamp'], inplace=True)

    # Map token to ID
    token2id = {t: i for i, t in enumerate(df['token_id'].unique())}
    id2token = {i: t for t, i in token2id.items()}
    df['token_id'] = df['token_id'].map(token2id)

    # Normalize OHLCV per token and store scalers
    scaled_data = []
    scalers = {}
    for token, group in df.groupby('token_id'):
        scaler = MinMaxScaler()
        features = scaler.fit_transform(group[['open', 'high', 'low', 'close', 'volume']])
        temp = pd.DataFrame(features, columns=['open', 'high', 'low', 'close', 'volume'])
        temp['token_id'] = token
        scaled_data.append(temp)
        scalers[token] = scaler

    df_scaled = pd.concat(scaled_data).reset_index(drop=True)
    return df, df_scaled, token2id, id2token, scalers


# --- Dataset ---
class OHLCVDataset(Dataset):
    def __init__(self, raw_df, token_id_col='token_id'):
        self.sequences = []
        self.targets = []

        tokens = raw_df[token_id_col].unique()
        for token in tokens:
            token_df = raw_df[raw_df[token_id_col] == token].reset_index(drop=True)
            data = token_df[['open', 'high', 'low', 'close', 'volume']].values
            token_id = token
            for i in range(len(data) - SEQUENCE_LENGTH):
                x = data[i:i+SEQUENCE_LENGTH]
                y = data[i+SEQUENCE_LENGTH][3] 
                self.sequences.append((x, token_id))
                self.targets.append(y)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, token_id = self.sequences[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        token_tensor = torch.tensor(token_id, dtype=torch.long)
        y_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x_tensor, token_tensor, y_tensor


# --- Transformer Model ---
class PriceTransformer(nn.Module):
    def __init__(self, num_tokens, input_dim=5, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.input_embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x, token_id):
        # Get sequence length from input
        batch_size, seq_len, _ = x.shape

        # Create token embedding and expand it to match sequence length
        token_embed = self.token_embedding(token_id)  
        token_embed = token_embed.unsqueeze(1)
        token_embed = token_embed.expand(batch_size, seq_len, -1) 

        # Process input through embedding layer
        x = self.input_embedding(x) 

        # Add token embedding to input embedding
        x = x + token_embed

        # Pass through transformer
        x = self.transformer(x) 
        x = x[:, -1, :] 
        return self.fc_out(x).squeeze()


# --- Training Function ---
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, token_id, y in dataloader:
        x, token_id, y = x.to(device), token_id.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, token_id)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- Prediction Function for All Tokens ---
def predict_all_tokens(model, df, scaled_df, token2id, id2token, scalers, device):
    model.eval()
    predictions = {}

    for token_id in token2id.values():
        token_df = scaled_df[scaled_df['token_id'] == token_id].reset_index(drop=True)

        # Skip tokens with insufficient data
        if len(token_df) <= SEQUENCE_LENGTH:
            print(f"Token ID {token_id} has insufficient data")
            continue

        # Get the last sequence for prediction
        last_sequence = token_df[['open', 'high', 'low', 'close', 'volume']].values[-SEQUENCE_LENGTH:]

        # Convert to tensor
        x = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)  
        token_tensor = torch.tensor([token_id], dtype=torch.long).to(device) 

        # Make prediction
        with torch.no_grad():
            pred = model(x, token_tensor).item()

        # Inverse transform the prediction
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred

        # Inverse transform
        original_scale_pred = scalers[token_id].inverse_transform(dummy)[0, 3]

        # Get the original token name
        original_token = id2token[token_id]
        predictions[original_token] = original_scale_pred

    return predictions


# --- Evaluation Function ---
def evaluate_predictions(model, df, scaled_df, token2id, id2token, scalers, device):
    model.eval()
    results = {}

    for token_id in token2id.values():
        token_df = scaled_df[scaled_df['token_id'] == token_id].reset_index(drop=True)

        # Skip tokens with insufficient data
        if len(token_df) <= SEQUENCE_LENGTH + 1: 
            print(f"Token ID {token_id} has insufficient data for evaluation")
            continue

        # Get the second-to-last sequence for prediction
        test_sequence = token_df[['open', 'high', 'low', 'close', 'volume']].values[-(SEQUENCE_LENGTH+1):-1]

        # Get the actual last price for comparison
        actual_price_normalized = token_df['close'].values[-1]

        # Convert dummy array to get the actual price in original scale
        dummy = np.zeros((1, 5))
        dummy[0, 3] = actual_price_normalized
        actual_price = scalers[token_id].inverse_transform(dummy)[0, 3]

        # Convert to tensor for prediction
        x = torch.tensor(test_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        token_tensor = torch.tensor([token_id], dtype=torch.long).to(device)

        # Make prediction
        with torch.no_grad():
            pred_normalized = model(x, token_tensor).item()

        # Inverse transform the prediction
        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred_normalized
        predicted_price = scalers[token_id].inverse_transform(dummy)[0, 3]

        # Calculate error metrics
        absolute_error = abs(predicted_price - actual_price)
        percentage_error = (absolute_error / actual_price) * 100 if actual_price != 0 else float('inf')

        # Get the original token name
        original_token = id2token[token_id]

        # Store results
        results[original_token] = {
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'absolute_error': absolute_error,
            'percentage_error': percentage_error
        }

    return results


# --- Compare Predictions with Real Prices ---
def compare_predictions_with_real(model, raw_df, scaled_df, token2id, id2token, scalers, device, test_size=5):
    """
    Compare predicted prices with real prices for the last test_size days
    """
    model.eval()
    comparison_results = {}

    for token_id in token2id.values():
        token_df = scaled_df[scaled_df['token_id'] == token_id].reset_index(drop=True)

        # Skip tokens with insufficient data
        if len(token_df) <= SEQUENCE_LENGTH + test_size:
            print(f"Token ID {token_id} has insufficient data for comparison")
            continue

        token_comparison = []

        # For each of the last test_size days
        for i in range(test_size):
            # Get the sequence ending test_size-i days before the end
            end_idx = len(token_df) - test_size + i
            test_sequence = token_df[['open', 'high', 'low', 'close', 'volume']].values[end_idx-SEQUENCE_LENGTH:end_idx]

            # Get the actual price for the day after the sequence
            actual_price_normalized = token_df['close'].values[end_idx]

            # Convert to original scale
            dummy = np.zeros((1, 5))
            dummy[0, 3] = actual_price_normalized
            actual_price = scalers[token_id].inverse_transform(dummy)[0, 3]

            # Convert to tensor for prediction
            x = torch.tensor(test_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            token_tensor = torch.tensor([token_id], dtype=torch.long).to(device)

            # Make prediction
            with torch.no_grad():
                pred_normalized = model(x, token_tensor).item()

            # Inverse transform the prediction
            dummy = np.zeros((1, 5))
            dummy[0, 3] = pred_normalized
            predicted_price = scalers[token_id].inverse_transform(dummy)[0, 3]

            # Calculate error metrics
            absolute_error = abs(predicted_price - actual_price)
            percentage_error = (absolute_error / actual_price) * 100 if actual_price != 0 else float('inf')

            # Get the date for this prediction
            date_idx = raw_df[(raw_df['token_id'] == token_id)].index[end_idx]
            date = raw_df.loc[date_idx, 'timestamp']

            # Store results for this day
            token_comparison.append({
                'date': date,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error
            })

        # Get the original token name
        original_token = id2token[token_id]
        comparison_results[original_token] = token_comparison

    return comparison_results


# --- Main Execution ---
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv(filepath)
    print(df.columns)
    raw_df, scaled_df, token2id, id2token, scalers = load_and_preprocess(filepath)

    # Create dataset and dataloaders
    dataset = OHLCVDataset(scaled_df)
    train_set, val_set = train_test_split(dataset, test_size=0.2, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PriceTransformer(num_tokens=len(token2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

    # Quick validation prediction
    model.eval()
    with torch.no_grad():
        for x, token_id, y in val_loader:
            x, token_id = x.to(device), token_id.to(device)
            preds = model(x, token_id)
            print("Predicted Close:", preds[0].item(), "True Close:", y[0].item())
            break

    # Evaluate model on the most recent data point
    print("\nEvaluating model performance...")
    evaluation_results = evaluate_predictions(model, raw_df, scaled_df, token2id, id2token, scalers, device)

    # Calculate average metrics
    total_abs_error = 0
    total_pct_error = 0
    valid_tokens = 0

    print("\nEvaluation Results:")
    print("Token\t\tPredicted\tActual\t\tAbs Error\tPct Error")
    print("-" * 80)

    for token, metrics in evaluation_results.items():
        print(f"{token[:10]:<10}\t${metrics['predicted_price']:<10.4f}\t${metrics['actual_price']:<10.4f}\t${metrics['absolute_error']:<10.4f}\t{metrics['percentage_error']:.2f}%")

        total_abs_error += metrics['absolute_error']
        if metrics['percentage_error'] != float('inf'):
            total_pct_error += metrics['percentage_error']
            valid_tokens += 1

    # Calculate and display average metrics
    avg_abs_error = total_abs_error / len(evaluation_results) if evaluation_results else 0
    avg_pct_error = total_pct_error / valid_tokens if valid_tokens > 0 else float('inf')

    print("-" * 80)
    print(f"Average Absolute Error: ${avg_abs_error:.4f}")
    print(f"Average Percentage Error: {avg_pct_error:.2f}%")

    # Compare predictions with real prices for the last 5 days
    print("\nComparing predictions with real prices for the last 5 days...")
    comparison_results = compare_predictions_with_real(model, raw_df, scaled_df, token2id, id2token, scalers, device, test_size=5)

    # Display detailed comparison results
    print("\nDetailed Comparison Results:")
    for token, days in comparison_results.items():
        print(f"\nToken: {token}")
        print("Date\t\tPredicted\tActual\t\tAbs Error\tPct Error")
        print("-" * 80)

        token_avg_pct_error = 0
        for day in days:
            date_str = day['date'].strftime('%Y-%m-%d')
            print(f"{date_str}\t${day['predicted_price']:<10.4f}\t${day['actual_price']:<10.4f}\t${day['absolute_error']:<10.4f}\t{day['percentage_error']:.2f}%")
            token_avg_pct_error += day['percentage_error']

        token_avg_pct_error /= len(days)
        print(f"Average Percentage Error for {token}: {token_avg_pct_error:.2f}%")

    # Calculate overall average error across all tokens and days
    all_pct_errors = []
    for token, days in comparison_results.items():
        for day in days:
            if day['percentage_error'] != float('inf'):
                all_pct_errors.append(day['percentage_error'])

    overall_avg_pct_error = sum(all_pct_errors) / len(all_pct_errors) if all_pct_errors else float('inf')
    print(f"\nOverall Average Percentage Error across all tokens and days: {overall_avg_pct_error:.2f}%")

    # Predict prices for all 50 tokens
    all_predictions = predict_all_tokens(model, raw_df, scaled_df, token2id, id2token, scalers, device)

    # Get the latest real prices for comparison
    latest_real_prices = {}
    for token_id, token_name in id2token.items():
        token_df = scaled_df[scaled_df['token_id'] == token_id]
        if len(token_df) > 0:
            # Get the latest normalized price
            latest_price_normalized = token_df['close'].values[-1]

            # Convert to original scale
            dummy = np.zeros((1, 5))
            dummy[0, 3] = latest_price_normalized
            latest_price = scalers[token_id].inverse_transform(dummy)[0, 3]

            latest_real_prices[token_name] = latest_price

    # Create a comprehensive DataFrame with predictions, real prices, and error metrics
    comparison_data = []
    for token, predicted_price in all_predictions.items():
        real_price = latest_real_prices.get(token, None)

        if real_price is not None:
            absolute_error = abs(predicted_price - real_price)
            percentage_error = (absolute_error / real_price) * 100 if real_price != 0 else float('inf')

            comparison_data.append({
                'token_id': token,
                'predicted_price': predicted_price,
                'real_price': real_price,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error
            })
        else:
            comparison_data.append({
                'token_id': token,
                'predicted_price': predicted_price,
                'real_price': 'N/A',
                'absolute_error': 'N/A',
                'percentage_error': 'N/A'
            })

    # Create and save the comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('token_price_comparison.csv', index=False)
    print("\nPredictions and comparisons saved to 'token_price_comparison.csv'")

    # Also save the detailed day-by-day comparison
    detailed_comparison_data = []
    for token, days in comparison_results.items():
        for day in days:
            detailed_comparison_data.append({
                'token_id': token,
                'date': day['date'],
                'predicted_price': day['predicted_price'],
                'actual_price': day['actual_price'],
                'absolute_error': day['absolute_error'],
                'percentage_error': day['percentage_error']
            })

    detailed_df = pd.DataFrame(detailed_comparison_data)
    detailed_df.to_csv('token_price_detailed_comparison.csv', index=False)
    print("Detailed day-by-day comparison saved to 'token_price_detailed_comparison.csv'")

    # Display predictions
    print("\nPredicted prices for all tokens:")
    for token, price in all_predictions.items():
        real_price = latest_real_prices.get(token, "N/A")
        if real_price != "N/A":
            error = abs(price - real_price)
            pct_error = (error / real_price) * 100 if real_price != 0 else float('inf')
            print(f"Token: {token}, Predicted: ${price:.4f}, Real: ${real_price:.4f}, Error: {pct_error:.2f}%")
        else:
            print(f"Token: {token}, Predicted: ${price:.4f}, Real: N/A")
