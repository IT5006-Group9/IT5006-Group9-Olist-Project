"""Frozen Phase 3 regression design. No test-driven configuration changes."""
from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'regression'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
PRIVATE = ROOT / 'data' / 'regression'
MODELS = ROOT / 'models' / 'regression'
SEED = 5006
START = pd.Timestamp('2017-01-01')
SELECTION_CUTOFF = TEST_START = pd.Timestamp('2018-06-01')
TEST_END = pd.Timestamp('2018-09-01')
MONTHS = ['2018-01','2018-02','2018-03','2018-04','2018-05']
ARCHIVE_SHA = '90ee50730a9e799aa2d3c7b1758fef680cbbc366f61a287870144796a37156d4'
FILES = {'orders':'olist_orders_dataset.csv','items':'olist_order_items_dataset.csv',
         'customers':'olist_customers_dataset.csv','sellers':'olist_sellers_dataset.csv',
         'products':'olist_products_dataset.csv','geo':'olist_geolocation_dataset.csv',
         'translation':'product_category_name_translation.csv'}
NUMERIC_GROUPS = {
 'promise':['promised_lead_time_days'],
 'geography':['max_distance_km','any_distance_missing','same_state'],
 'order':['total_price','total_freight','freight_ratio','item_count','seller_count',
          'primary_seller_price_share','seller_state_count'],
 'product':['avg_product_weight_g','any_weight_missing','distinct_product_count',
            'category_count_observed','dominant_category_price_share','max_product_weight_g',
            'total_known_weight_g','max_product_volume_cm3','total_known_volume_cm3',
            'mean_unit_price','any_volume_missing'],
 'calendar':['month_sin','month_cos','weekday_sin','weekday_cos','hour_sin','hour_cos'],
}
CAT_GROUPS = {
 'geography':['customer_state','route_group','primary_seller_state','primary_seller_city',
              'customer_city','primary_state_route','primary_city_route','primary_zip_zone_route'],
 'seller':['primary_seller_id'], 'product':['category_group','dominant_category']}
FORBIDDEN = ['delivered_ts','lead_time_days','order_status','review_score','review_answer_timestamp',
             'order_delivered_carrier_date','order_approved_at','order_delivered_customer_date']
SUCCESS = {'mae_reduction_vs_training_median':0.05,'beat_delivery_promise':True}

def columns(pack='calendar'):
    numeric=[c for g,cs in NUMERIC_GROUPS.items() if g!='calendar' or pack!='no_calendar' for c in cs]
    if pack=='absolute': numeric += ['purchase_elapsed_days']
    if pack=='risk': numeric += ['risk_p30','risk_unavailable']
    categorical=[c for cs in CAT_GROUPS.values() for c in cs]
    if pack=='absolute': categorical += ['purchase_year_month']
    return numeric,categorical

def directories():
    for p in (OUT,TABLES,FIGURES,PRIVATE,MODELS,OUT/'predictions'):
        p.mkdir(parents=True,exist_ok=True)
