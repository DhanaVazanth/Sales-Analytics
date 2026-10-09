import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

DATASET_DIR = "/Users/dhanavasanth/Downloads/sales_analytics_project/dataset"
OUTPUT_JSON = "/Users/dhanavasanth/Downloads/sales_analytics_project/predictions_data.json"

def load_data():
    print("Loading dataset CSV files...")
    customers = pd.read_csv(os.path.join(DATASET_DIR, "customers.csv"))
    plans = pd.read_csv(os.path.join(DATASET_DIR, "plans.csv"))
    subscriptions = pd.read_csv(os.path.join(DATASET_DIR, "subscriptions.csv"))
    recharges = pd.read_csv(os.path.join(DATASET_DIR, "recharges.csv"))
    call_records = pd.read_csv(os.path.join(DATASET_DIR, "call_records.csv"))
    data_usage = pd.read_csv(os.path.join(DATASET_DIR, "data_usage.csv"))
    return customers, plans, subscriptions, recharges, call_records, data_usage

def run_ml_pipeline():
    customers, plans, subscriptions, recharges, call_records, data_usage = load_data()

    # Data processing & Feature Aggregations per customer
    print("Processing customer features...")
    
    # Recharges aggregation per customer
    recharges['recharge_date'] = pd.to_datetime(recharges['recharge_date'])
    max_recharge_date = recharges['recharge_date'].max()
    
    rec_agg = recharges.groupby('customer_id').agg(
        total_recharge_amt=('amount', 'sum'),
        recharge_count=('recharge_id', 'count'),
        last_recharge_date=('recharge_date', 'max'),
        avg_recharge_amt=('amount', 'mean')
    ).reset_index()
    
    rec_agg['recency_days'] = (max_recharge_date - rec_agg['last_recharge_date']).dt.days

    # Call Records aggregation per customer
    call_records['call_date'] = pd.to_datetime(call_records['call_date'])
    call_agg = call_records.groupby('customer_id').agg(
        total_call_duration=('call_duration_sec', 'sum'),
        call_count=('call_id', 'count'),
        roaming_call_count=('call_type', lambda x: (x == 'Roaming').sum())
    ).reset_index()
    call_agg['roaming_ratio'] = np.where(call_agg['call_count'] > 0, call_agg['roaming_call_count'] / call_agg['call_count'], 0)

    # Data Usage aggregation per customer
    data_usage['usage_date'] = pd.to_datetime(data_usage['usage_date'])
    data_agg = data_usage.groupby('customer_id').agg(
        total_data_mb=('data_used_mb', 'sum'),
        avg_session_mins=('session_time_mins', 'mean'),
        data_session_count=('usage_id', 'count')
    ).reset_index()

    # Latest active subscription plan mapping
    subscriptions['start_date'] = pd.to_datetime(subscriptions['start_date'])
    latest_sub = subscriptions.sort_values('start_date').groupby('customer_id').last().reset_index()
    latest_sub = latest_sub.merge(plans, on='plan_id', how='left')

    # Master Customer Feature Table
    df = customers.merge(rec_agg, on='customer_id', how='left')
    df = df.merge(call_agg, on='customer_id', how='left')
    df = df.merge(data_agg, on='customer_id', how='left')
    df = df.merge(latest_sub[['customer_id', 'plan_id', 'plan_name', 'plan_type', 'price', 'data_gb_per_day', 'validity_days']], on='customer_id', how='left')

    # Fill Missing values
    df['total_recharge_amt'] = df['total_recharge_amt'].fillna(0)
    df['recharge_count'] = df['recharge_count'].fillna(0)
    df['avg_recharge_amt'] = df['avg_recharge_amt'].fillna(0)
    df['recency_days'] = df['recency_days'].fillna(180)
    df['total_call_duration'] = df['total_call_duration'].fillna(0)
    df['call_count'] = df['call_count'].fillna(0)
    df['roaming_ratio'] = df['roaming_ratio'].fillna(0)
    df['total_data_mb'] = df['total_data_mb'].fillna(0)
    df['avg_session_mins'] = df['avg_session_mins'].fillna(0)
    df['price'] = df['price'].fillna(299)
    df['data_gb_per_day'] = df['data_gb_per_day'].fillna(1.5)

    # Calculate Target: Churn Flag (1 if Inactive, else 0)
    df['is_churn'] = np.where(df['status'] == 'Inactive', 1, 0)

    # RFM Scoring calculation
    print("Calculating RFM Scores...")
    # Recency score (reverse percentile: lower days = higher score)
    df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    # Frequency score
    df['f_score'] = pd.qcut(df['recharge_count'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary score
    df['m_score'] = pd.qcut(df['total_recharge_amt'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['rfm_combined'] = df['r_score'].astype(str) + df['f_score'].astype(str) + df['m_score'].astype(str)

    def assign_segment(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions / VIP"
        elif r >= 3 and m >= 4:
            return "Potential Loyalists"
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At-Risk High Value"
        else:
            return "Budget / Light Users"

    df['rfm_segment'] = df.apply(assign_segment, axis=1)

    # Train Model 1: Churn Classifier
    print("Training Churn Risk Model...")
    feature_cols = ['recency_days', 'recharge_count', 'total_recharge_amt', 'total_call_duration', 'roaming_ratio', 'total_data_mb', 'price', 'data_gb_per_day']
    X = df[feature_cols]
    y = df['is_churn']

    model_churn = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    model_churn.fit(X, y)

    # Predict Churn Probabilities
    df['churn_probability'] = model_churn.predict_proba(X)[:, 1]
    
    # Feature Importances
    feature_importances = dict(zip(feature_cols, [round(float(x), 4) for x in model_churn.feature_importances_]))

    # Train Model 2: Expected Recharge Amount Predictor
    print("Training Expected Recharge Amount Predictor...")
    model_recharge = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    model_recharge.fit(X, df['avg_recharge_amt'])
    df['predicted_next_recharge'] = model_recharge.predict(X).round(2)

    # Model 3: Subscriber Growth Forecasting (Next 90 Days Time Series)
    print("Generating Subscriber Growth & Churn Forecast...")
    # Generate historical daily dynamic active base trend
    base_active = int((df['status'] == 'Active').sum())
    forecast_days = 90
    dates = [(datetime(2026, 10, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(forecast_days)]
    
    # Simulate time series trend using baseline + growth - churn projections
    trend_base = []
    curr_active = base_active
    np.random.seed(42)
    
    for i in range(forecast_days):
        daily_joins = int(np.random.normal(loc=18, scale=4))
        daily_churns = int(np.random.normal(loc=12, scale=3))
        curr_active = curr_active + daily_joins - daily_churns
        trend_base.append(curr_active)
        
    subscriber_forecast = {
        "dates": dates,
        "historical_last_active": base_active,
        "projected_active": trend_base,
        "upper_bound": [int(x * 1.035) for x in trend_base],
        "lower_bound": [int(x * 0.965) for x in trend_base],
    }

    # Model 4: Network Data Usage Demand Forecasting by Circle
    print("Generating Network Data Usage Demand Forecast...")
    circle_demand = data_usage.merge(customers[['customer_id', 'circle']], on='customer_id', how='left')
    circle_summary = circle_demand.groupby('circle').agg(
        current_daily_gb=('data_used_mb', lambda x: round(x.sum() / 1024, 2)),
        avg_session_mins=('session_time_mins', 'mean')
    ).reset_index()

    # Forecast 30-day demand growth per circle (assuming 8% monthly growth rate in data demand)
    circle_summary['projected_monthly_tb'] = (circle_summary['current_daily_gb'] * 30 * 1.08 / 1024).round(2)
    circle_summary['capacity_status'] = np.where(circle_summary['projected_monthly_tb'] > 120, 'High Load / Action Required', 'Optimal Capacity')

    # Model 5: Next-Best-Offer (NBO) Plan Recommender
    print("Generating Next-Best-Offer Plan Recommendations...")
    plans_dict = plans.to_dict(orient='records')
    
    def get_nbo(row):
        daily_mb = row['total_data_mb'] / 30.0 if row['total_data_mb'] > 0 else 500
        monthly_spend = row['total_recharge_amt']
        
        # High data user upgrade rule
        if daily_mb > 1500 and monthly_spend < 500:
            return {"recommended_plan_id": "P001", "recommended_plan_name": "Unlimited 5G Data (₹299)", "reason": "High data consumption (upgrade for speed)"}
        elif row['churn_probability'] > 0.6:
            return {"recommended_plan_id": "P002", "recommended_plan_name": "Truly Unlimited (₹479 - 20% OFF Winback)", "reason": "High churn risk retention offer"}
        elif monthly_spend > 800:
            return {"recommended_plan_id": "P003", "recommended_plan_name": "Annual Plan (₹2999)", "reason": "VIP Spender long-tenure locking"}
        elif row['total_call_duration'] > 1500 and daily_mb < 200:
            return {"recommended_plan_id": "P005", "recommended_plan_name": "Talktime Only (₹99)", "reason": "Voice heavy user profile"}
        else:
            return {"recommended_plan_id": "P010", "recommended_plan_name": "Entertainment Pack (₹349)", "reason": "Balanced data + voice OTT value pack"}

    nbo_results = df.apply(get_nbo, axis=1)
    df['nbo_plan_id'] = [n['recommended_plan_id'] for n in nbo_results]
    df['nbo_plan_name'] = [n['recommended_plan_name'] for n in nbo_results]
    df['nbo_reason'] = [n['reason'] for n in nbo_results]

    # Overall Summary Statistics
    summary_stats = {
        "total_customers": int(len(df)),
        "active_customers": int((df['status'] == 'Active').sum()),
        "inactive_customers": int((df['status'] == 'Inactive').sum()),
        "overall_churn_rate_pct": round(float((df['status'] == 'Inactive').mean() * 100), 2),
        "avg_churn_probability": round(float(df['churn_probability'].mean() * 100), 2),
        "total_revenue": round(float(df['total_recharge_amt'].sum()), 2),
        "avg_arpu": round(float(df['total_recharge_amt'].mean()), 2),
        "total_data_tb": round(float(df['total_data_mb'].sum() / (1024 * 1024)), 2),
        "high_risk_churn_count": int((df['churn_probability'] > 0.65).sum()),
        "rfm_segment_counts": df['rfm_segment'].value_counts().to_dict()
    }

    # Format customer records for JSON
    customer_records = []
    for _, row in df.iterrows():
        customer_records.append({
            "customer_id": str(row['customer_id']),
            "name": str(row['name']),
            "phone_number": str(row['phone_number']),
            "city": str(row['city']),
            "circle": str(row['circle']),
            "join_date": str(row['join_date']),
            "status": str(row['status']),
            "recency_days": int(row['recency_days']),
            "recharge_count": int(row['recharge_count']),
            "total_recharge_amt": float(row['total_recharge_amt']),
            "total_data_mb": float(row['total_data_mb']),
            "total_call_duration": int(row['total_call_duration']),
            "churn_probability": round(float(row['churn_probability'] * 100), 1),
            "risk_level": "High" if row['churn_probability'] > 0.65 else ("Medium" if row['churn_probability'] > 0.35 else "Low"),
            "rfm_segment": str(row['rfm_segment']),
            "current_plan": str(row['plan_name']) if pd.notnull(row['plan_name']) else "Basic Prepaid",
            "predicted_next_recharge": float(row['predicted_next_recharge']),
            "nbo_plan_name": str(row['nbo_plan_name']),
            "nbo_reason": str(row['nbo_reason'])
        })

    output_payload = {
        "summary": summary_stats,
        "feature_importances": feature_importances,
        "subscriber_forecast": subscriber_forecast,
        "circle_demand": circle_summary.to_dict(orient='records'),
        "plans_catalog": plans_dict,
        "customers": customer_records
    }

    print(f"Saving predictions output to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_payload, f, indent=2)

    print("✅ ML Engine Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_ml_pipeline()
