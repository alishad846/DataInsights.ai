import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_erp_sales_data(rows=1500):
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    
    # Base Categoricals
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    countries = {'North America': ['USA', 'Canada'], 'Europe': ['UK', 'Germany', 'France'], 
                 'Asia Pacific': ['Japan', 'Australia', 'China'], 'Latin America': ['Brazil', 'Mexico']}
    categories = ['Electronics', 'Furniture', 'Office Supplies', 'Software']
    brands = ['Apple', 'Dell', 'Sony', 'Samsung', 'Logitech', 'IKEA', 'Herman Miller', 'Microsoft']
    sales_channels = ['Online', 'Retail', 'Direct Sales', 'Partner']
    campaign_sources = ['Google Ads', 'Organic', 'Email', 'Referral', 'None']
    payment_methods = ['Credit Card', 'PayPal', 'Bank Transfer', 'Stripe']
    order_statuses = ['Delivered', 'Shipped', 'Processing']
    warehouses = ['WH-East', 'WH-West', 'WH-Central', 'WH-Europe', 'WH-Asia']
    
    data = []
    
    for i in range(rows):
        order_date = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        delivery_date = order_date + timedelta(days=random.randint(2, 14))
        
        region = random.choice(regions)
        country = random.choice(countries[region])
        category = random.choice(categories)
        brand = random.choice(brands)
        
        qty = random.randint(1, 100)
        unit_price = round(random.uniform(10.0, 1500.0), 2)
        gross_sales = round(qty * unit_price, 2)
        discount_pct = random.choice([0, 0, 0.05, 0.10, 0.15, 0.20])
        discount_amt = round(gross_sales * discount_pct, 2)
        net_sales = gross_sales - discount_amt
        tax = round(net_sales * 0.08, 2)
        shipping_cost = round(random.uniform(5.0, 50.0), 2) if net_sales < 1000 else 0.0
        revenue = net_sales + tax + shipping_cost
        
        cost_margin = random.uniform(0.4, 0.8)
        cost = round(gross_sales * cost_margin, 2)
        profit = round(net_sales - cost - shipping_cost, 2)
        
        row = {
            'order_id': f'ORD-{10000+i}',
            'order_reference': f'REF-2023-{random.randint(100000, 999999)}',
            'order_date': order_date.strftime('%Y-%m-%d %H:%M:%S'),
            'delivery_date': delivery_date.strftime('%Y-%m-%d'),
            'customer_id': f'CUST-{random.randint(100, 999)}',
            'customer_name': f'Customer_{random.randint(1, 500)}',
            'company': f'Company_{random.randint(1, 200)} LLC',
            'salesperson': f'Rep_{random.randint(1, 20)}',
            'region': region,
            'country': country,
            'city': f'{country}_City',
            'product_id': f'PRD-{random.randint(1000, 5000)}',
            'product_name': f'{brand} {category} Model {random.randint(1, 100)}',
            'product_category': category,
            'product_brand': brand,
            'quantity': qty,
            'unit_price': unit_price,
            'gross_sales': gross_sales,
            'discount': discount_amt,
            'net_sales': net_sales,
            'tax': tax,
            'cost': cost,
            'revenue': revenue,
            'profit': profit,
            'profit_margin_pct': round((profit / revenue) * 100, 2) if revenue > 0 else 0,
            'shipping_cost': shipping_cost,
            'payment_method': random.choice(payment_methods),
            'order_status': random.choices(order_statuses, weights=[0.8, 0.15, 0.05])[0],
            'warehouse': random.choice(warehouses),
            'currency': 'USD',
            'exchange_rate': 1.0,
            'sales_channel': random.choice(sales_channels),
            'campaign_source': random.choice(campaign_sources),
            'created_at': (order_date - timedelta(minutes=random.randint(1, 60))).strftime('%Y-%m-%d %H:%M:%S')
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Introduce a few nulls for realism
    for _ in range(30):
        df.loc[random.randint(0, rows-1), 'discount'] = np.nan
        df.loc[random.randint(0, rows-1), 'campaign_source'] = np.nan
        
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realistic_erp_sales.csv')
    df.to_csv(out_path, index=False)
    print(f'Generated dataset with {df.shape[0]} rows and {df.shape[1]} columns at {out_path}')

if __name__ == '__main__':
    generate_erp_sales_data(1500)
