import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache("f1_cache")

# CONFIGURATION - DIFFERENT FOR EACH RACE
RACE_YEAR = 2025
RACE_NAME = "United States Grand Prix"
RACE_NUMBER = 19
         
# 2025 Qualifying Data 
qualifying_2025 = pd.DataFrame({
  
  "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell",
             "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
             "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
  "QualifyingTime (s)": [92.801, 93.084, 92.510, 92.826,
                         0,        0,        92.807,  92.912,
                         0,        0,        93.160,  0]


})

# Driver mapping (3-letter codes)
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", 
    "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", 
    "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS", 
    "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO",
    "Sergio Perez": "PER", "Nico Hulkenberg": "HUL", "Kevin Magnussen": "MAG",
    "Esteban Ocon": "OCO", "Oliver Bearman": "BEA", "Jack Doohan": "DOO",
    "Isack Hadjar": "HAD", "Gabriel Bortoleto": "BOR"
}

# LOAD 2024 TRAINING DATA
print(f"Loading 2024 Round {RACE_NUMBER} ({RACE_NAME}) data for training...")
print("This may take a few minutes on first run...")

try:
    # Try loading by round number first (more reliable)
    session_2024 = fastf1.get_session(2024, RACE_NUMBER, "R")
    session_2024.load(telemetry=False, weather=False, messages=False)
    
    # Verify data is loaded
    if not hasattr(session_2024, '_laps') or session_2024._laps is None:
        raise ValueError("Session data failed to load properly")
    
    # Extract lap times from 2024
    laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
    laps_2024.dropna(subset=["LapTime"], inplace=True)
    laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()
    
    # Remove invalid lap times (0 or negative)
    laps_2024 = laps_2024[laps_2024["LapTime (s)"] > 0]
    
    if len(laps_2024) == 0:
        raise ValueError("No valid lap times found in 2024 data")
    
    print(f"✓ Loaded {len(laps_2024)} laps from 2024 {RACE_NAME}")
    
except Exception as e:
    print(f"\n❌ Error loading 2024 data: {e}")
    print(f"\nTroubleshooting:")
    print(f"1. Check if race round {RACE_NUMBER} exists in 2024")
    print(f"2. Verify your internet connection")
    print(f"3. Clear cache: delete 'f1_cache' folder and try again")
    print(f"4. Try updating fastf1: pip install --upgrade fastf1")
    raise

# PREPARE 2025 QUALIFYING DATA
qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Filter out drivers not in qualifying
qualifying_2025 = qualifying_2025.dropna(subset=["DriverCode"])

# MERGE TRAINING DATA
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

if merged_data.shape[0] == 0:
    raise ValueError("No matching drivers found between 2024 and 2025 data!")

print(f"✓ Matched {len(merged_data)} data points for training")

# TRAIN MODEL
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

print("\nTraining Gradient Boosting Model...")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"✓ Model trained successfully!")

# MAKE PREDICTIONS FOR 2025
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)
qualifying_2025["Position"] = range(1, len(qualifying_2025) + 1)

# DISPLAY RESULTS
print("\n" + "="*60)
print(f"🏁 Predicted {RACE_YEAR} {RACE_NAME} Winner 🏁")
print("="*60)

winner = qualifying_2025.iloc[0]
print(f"\nDriver: {winner['Driver']}, Predicted Race Time: {winner['PredictedRaceTime (s)']:.2f}s")

print("\nTop 3:")
for i in range(min(3, len(qualifying_2025))):
    driver = qualifying_2025.iloc[i]
    print(f"{i+1}. {driver['Driver']} - {driver['PredictedRaceTime (s)']:.2f}s (Qualified: {driver['QualifyingTime (s)']:.3f}s)")

print("\n" + "="*60)
print(f"Model Error (MAE): {mae:.2f} seconds")
print("="*60)

print("\nFull Predicted Results:")
print(qualifying_2025[["Position", "Driver", "QualifyingTime (s)", "PredictedRaceTime (s)"]].to_string(index=False))