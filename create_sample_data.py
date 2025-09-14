import pandas as pd
import numpy as np

# Create comprehensive sample appliance dataset
np.random.seed(42)

# Common Indian household appliances with realistic specifications
appliances_data = []

# Define appliance categories with typical specifications
appliance_specs = {
    'Refrigerator': {
        'power_range': (150, 300),
        'hours_range': (24, 24),  # Always on
        'star_ratings': [2, 3, 4, 5],
        'efficiency_factor': 0.8
    },
    'Air Conditioner': {
        'power_range': (1000, 2000),
        'hours_range': (6, 14),
        'star_ratings': [2, 3, 4, 5],
        'efficiency_factor': 0.9
    },
    'Television': {
        'power_range': (80, 200),
        'hours_range': (4, 10),
        'star_ratings': [3, 4, 5],
        'efficiency_factor': 0.95
    },
    'Washing Machine': {
        'power_range': (400, 800),
        'hours_range': (1, 3),
        'star_ratings': [3, 4, 5],
        'efficiency_factor': 0.85
    },
    'Water Heater': {
        'power_range': (1500, 3000),
        'hours_range': (2, 4),
        'star_ratings': [2, 3, 4, 5],
        'efficiency_factor': 0.75
    },
    'Microwave': {
        'power_range': (700, 1200),
        'hours_range': (0.5, 2),
        'star_ratings': [3, 4, 5],
        'efficiency_factor': 0.9
    },
    'Ceiling Fan': {
        'power_range': (50, 80),
        'hours_range': (8, 16),
        'star_ratings': [3, 4, 5],
        'efficiency_factor': 0.95
    },
    'LED Lights': {
        'power_range': (9, 20),
        'hours_range': (6, 12),
        'star_ratings': [4, 5],
        'efficiency_factor': 0.98
    },
    'Laptop': {
        'power_range': (45, 90),
        'hours_range': (6, 12),
        'star_ratings': [4, 5],
        'efficiency_factor': 0.9
    },
    'Desktop Computer': {
        'power_range': (200, 400),
        'hours_range': (4, 10),
        'star_ratings': [3, 4],
        'efficiency_factor': 0.85
    }
}

# Room size categories (sq ft)
room_sizes = {
    'Small': (80, 150),
    'Medium': (150, 250),
    'Large': (250, 400)
}

# Generate data for multiple households
households = []
household_id = 1

for _ in range(200):  # 200 different households
    household = {
        'household_id': household_id,
        'household_size': np.random.choice([2, 3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'location_type': np.random.choice(['Urban', 'Suburban', 'Rural'], p=[0.5, 0.3, 0.2]),
        'income_bracket': np.random.choice(['Low', 'Middle', 'High'], p=[0.3, 0.5, 0.2]),
        'house_type': np.random.choice(['Apartment', 'Independent House'], p=[0.6, 0.4]),
        'total_rooms': np.random.randint(2, 6),
        'monthly_bill': 0  # Will be calculated
    }
    households.append(household)
    household_id += 1

# Generate appliance data for each household
for household in households:
    # Determine how many appliances based on household characteristics
    base_appliances = ['Refrigerator', 'Television', 'LED Lights', 'Ceiling Fan']
    
    if household['income_bracket'] in ['Middle', 'High']:
        base_appliances.extend(['Air Conditioner', 'Washing Machine', 'Microwave'])
    
    if household['income_bracket'] == 'High':
        base_appliances.extend(['Water Heater', 'Laptop', 'Desktop Computer'])
    
    # Add seasonal variation
    season = np.random.choice(['Summer', 'Winter', 'Monsoon', 'Spring'])
    
    for appliance_type in base_appliances:
        if appliance_type in appliance_specs:
            specs = appliance_specs[appliance_type]
            
            # Random variations in specifications
            power_rating = np.random.randint(specs['power_range'][0], specs['power_range'][1] + 1)
            star_rating = np.random.choice(specs['star_ratings'])
            
            # Adjust hours based on season and household characteristics
            base_hours = np.random.uniform(specs['hours_range'][0], specs['hours_range'][1])
            
            # Seasonal adjustments
            if appliance_type == 'Air Conditioner' and season == 'Summer':
                base_hours *= 1.5
            elif appliance_type == 'Air Conditioner' and season == 'Winter':
                base_hours *= 0.3
            elif appliance_type == 'Water Heater' and season == 'Winter':
                base_hours *= 1.3
            
            # Household size adjustments
            if appliance_type in ['Washing Machine', 'Water Heater', 'Refrigerator']:
                base_hours *= (household['household_size'] / 4)
            
            daily_hours = max(0.5, min(24, base_hours))
            
            # Calculate room size
            room_category = np.random.choice(['Small', 'Medium', 'Large'], p=[0.4, 0.4, 0.2])
            room_size = np.random.randint(room_sizes[room_category][0], room_sizes[room_category][1] + 1)
            
            # Calculate efficiency factor based on star rating
            efficiency_multiplier = {5: 0.8, 4: 0.9, 3: 1.0, 2: 1.2, 1: 1.4}
            
            # Calculate daily consumption (kWh)
            base_consumption = (power_rating * daily_hours) / 1000
            actual_consumption = base_consumption * efficiency_multiplier[star_rating]
            
            # Add some random variation
            actual_consumption *= np.random.uniform(0.9, 1.1)
            
            # Calculate monthly consumption and cost
            monthly_consumption = actual_consumption * 30
            electricity_rate = 6.5  # Rs per kWh (average in India)
            monthly_cost = monthly_consumption * electricity_rate
            
            appliance_record = {
                'household_id': household['household_id'],
                'appliance_type': appliance_type,
                'power_rating': power_rating,
                'star_rating': star_rating,
                'daily_hours': round(daily_hours, 2),
                'room_size': room_size,
                'household_size': household['household_size'],
                'location_type': household['location_type'],
                'income_bracket': household['income_bracket'],
                'house_type': household['house_type'],
                'season': season,
                'daily_consumption': round(actual_consumption, 3),
                'monthly_consumption': round(monthly_consumption, 2),
                'monthly_cost': round(monthly_cost, 2),
                'efficiency_score': star_rating * 20,  # 0-100 scale
                'usage_pattern': 'Regular' if daily_hours <= 8 else 'Heavy',
                'appliance_age': np.random.randint(1, 10),  # years
                'brand_tier': np.random.choice(['Premium', 'Standard', 'Budget'], p=[0.2, 0.5, 0.3])
            }
            
            appliances_data.append(appliance_record)

# Create DataFrame
df = pd.DataFrame(appliances_data)

# Calculate total monthly bill for each household
household_bills = df.groupby('household_id')['monthly_cost'].sum().reset_index()
household_bills.columns = ['household_id', 'total_monthly_bill']

# Merge back to main dataframe
df = df.merge(household_bills, on='household_id')

# Add some derived features
df['cost_per_kwh'] = df['monthly_cost'] / df['monthly_consumption']
df['consumption_per_watt'] = df['daily_consumption'] / df['power_rating'] * 1000
df['efficiency_ratio'] = df['star_rating'] / (df['power_rating'] / 1000)

# Save the dataset
df.to_csv('data/sample/appliance_energy_dataset.csv', index=False)

print("Sample dataset created successfully!")
print(f"Total records: {len(df)}")
print(f"Unique households: {df['household_id'].nunique()}")
print(f"Appliance types: {df['appliance_type'].nunique()}")
print("\nDataset overview:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSample statistics:")
print(df.describe())