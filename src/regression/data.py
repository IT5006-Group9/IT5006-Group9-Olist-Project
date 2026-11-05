"""Source-only purchase features; target and outcome metadata are attached separately."""
import hashlib, json, io, os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from common.geography import geographic_reference, compute_routes
from . import config as C

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ids_hash(ids): return hashlib.sha256('\n'.join(sorted(map(str,ids))).encode()).hexdigest()
def save_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n')
def read_tables():
    archive=os.environ.get('OLIST_ARCHIVE')
    directory=os.environ.get('OLIST_CSV_DIR')
    if not archive:
        raise FileNotFoundError('Set OLIST_ARCHIVE to the fixed course ZIP. A CSV override also requires it for fingerprint verification.')
    inventory=[]; tables={}
    if archive and digest(archive)!=C.ARCHIVE_SHA:
        raise ValueError('The course archive fingerprint differs; do not substitute a newer dataset.')
    for k,name in C.FILES.items():
        if directory:
            content=(Path(directory)/name).read_bytes()
            with zipfile.ZipFile(archive) as z: expected=z.read('Olist_CSV/'+name)
            if hashlib.sha256(content).digest()!=hashlib.sha256(expected).digest():
                raise ValueError('CSV override differs from the fixed course archive: '+name)
        else:
            with zipfile.ZipFile(archive) as z: content=z.read('Olist_CSV/'+name)
        tables[k]=pd.read_csv(io.BytesIO(content))
        inventory.append({'table':k,'file':name,'rows':len(tables[k]),'sha256':hashlib.sha256(content).hexdigest()})
    return tables,pd.DataFrame(inventory)

def features(t):
    """One row per order, using only purchase/quote and archived product/location fields."""
    orders,items,customers,sellers,products=[t[k] for k in ['orders','items','customers','sellers','products']]
    for table,key in [(orders,'order_id'),(customers,'customer_id'),(sellers,'seller_id'),(products,'product_id')]:
        assert table[key].is_unique
    assert not items.duplicated(['order_id','order_item_id']).any()
    tr=t['translation'];assert tr.product_category_name.is_unique
    product=products.merge(tr,on='product_category_name',how='left',validate='many_to_one')
    # Outcome columns are deliberately absent from this join.
    o=orders[['order_id','customer_id','order_purchase_timestamp','order_estimated_delivery_date']].copy()
    o['purchase_ts']=pd.to_datetime(o.order_purchase_timestamp)
    o['estimated_ts']=pd.to_datetime(o.order_estimated_delivery_date)
    loc=o.merge(customers,on='customer_id',how='left',validate='many_to_one')
    it=items.merge(product,on='product_id',how='left',validate='many_to_one').merge(
        sellers[['seller_id','seller_state']],on='seller_id',how='left',validate='many_to_one')
    g=it.groupby('order_id',sort=True)
    a=g.agg(total_price=('price','sum'),total_freight=('freight_value','sum'),
        item_count=('order_item_id','size'),seller_count=('seller_id','nunique'),
        category_count=('product_category_name_english','nunique'),
        first_category=('product_category_name_english','first'),
        distinct_product_count=('product_id','nunique'),category_count_observed=('product_category_name','nunique'),
        seller_state_count=('seller_state','nunique'),mean_unit_price=('price','mean'))
    weight=it.product_weight_g.where(it.product_weight_g.gt(0))
    dims=it[['product_length_cm','product_height_cm','product_width_cm']]
    volume=dims.prod(axis=1).where(dims.notna().all(axis=1)&dims.gt(0).all(axis=1))
    it['weight']=weight;it['volume']=volume
    g=it.groupby('order_id',sort=True)
    a['any_weight_missing']=g.weight.count().lt(g.size()).astype(int)
    a['avg_product_weight_g']=g.weight.mean().mask(a.any_weight_missing.astype(bool))
    a['max_product_weight_g']=g.weight.max()
    a['total_known_weight_g']=g.weight.sum(min_count=1)
    a['max_product_volume_cm3']=g.volume.max()
    a['total_known_volume_cm3']=g.volume.sum(min_count=1)
    a['any_volume_missing']=g.volume.count().lt(g.size()).astype(int)
    missing=g.product_category_name_english.count().lt(g.size())
    a['category_group']=np.select([missing,a.category_count.gt(1)],
        ['Missing/incomplete category','Multiple categories'],default=a.first_category)
    a['freight_ratio']=a.total_freight/a.total_price.where(a.total_price.gt(0))
    geo=geographic_reference(t['geo'],'screened_median')
    route,dist=compute_routes(items,orders,customers,sellers,geo)
    same=route.customer_state.eq(route.seller_state).groupby(route.order_id).all()
    known=(route.customer_state.notna()&route.seller_state.notna()).groupby(route.order_id).all()
    a=a.join(dist);a['same_state']=same.astype(int)
    a['route_group']=np.select([~known,~same],['Unknown','Any seller cross-state'],default='All sellers same-state')
    primary=items.groupby(['order_id','seller_id'],as_index=False).price.sum().sort_values(
        ['order_id','price','seller_id'],ascending=[True,False,True]).drop_duplicates('order_id')
    primary=primary.merge(sellers,on='seller_id',how='left',validate='many_to_one').rename(
        columns={'seller_id':'primary_seller_id','seller_state':'primary_seller_state',
                 'seller_city':'primary_seller_city','price':'primary_seller_price'})
    x=loc.merge(a,on='order_id',how='inner',validate='one_to_one').merge(primary,on='order_id',how='left',validate='one_to_one')
    cat=it[['order_id','product_category_name','price']].copy()
    cat['product_category_name']=cat.product_category_name.fillna('__missing_category__')
    cat=cat.groupby(['order_id','product_category_name'],as_index=False).price.sum().sort_values(
        ['order_id','price','product_category_name'],ascending=[True,False,True]).drop_duplicates('order_id')
    cat=cat.rename(columns={'product_category_name':'dominant_category','price':'dominant_category_price'})
    x=x.merge(cat,on='order_id',how='left',validate='one_to_one')
    x['primary_seller_price_share']=x.primary_seller_price/x.total_price.where(x.total_price.gt(0))
    x['dominant_category_price_share']=x.dominant_category_price/x.total_price.where(x.total_price.gt(0))
    x['primary_state_route']=x.primary_seller_state+'|'+x.customer_state
    x['primary_city_route']=x.primary_seller_state+':'+x.primary_seller_city+'|'+x.customer_state+':'+x.customer_city
    x['primary_zip_zone_route']=x.seller_zip_code_prefix.floordiv(1000).astype(str)+'|'+x.customer_zip_code_prefix.floordiv(1000).astype(str)
    x['promised_lead_time_days']=(x.estimated_ts-x.purchase_ts).dt.total_seconds()/86400
    for name,v,period in [('month',x.purchase_ts.dt.month-1,12),('weekday',x.purchase_ts.dt.dayofweek,7),('hour',x.purchase_ts.dt.hour,24)]:
        x[name+'_sin']=np.sin(2*np.pi*v/period);x[name+'_cos']=np.cos(2*np.pi*v/period)
    x['purchase_elapsed_days']=(x.purchase_ts-pd.Timestamp('2016-01-01')).dt.total_seconds()/86400
    x['purchase_year_month']=x.purchase_ts.dt.strftime('%Y-%m')
    nums,cats=C.columns('absolute')
    x=x[['order_id','customer_unique_id','purchase_ts','estimated_ts',*nums,*cats]].sort_values(['purchase_ts','order_id']).reset_index(drop=True)
    assert x.order_id.is_unique
    assert not set(C.FORBIDDEN)&set(x.columns)
    return x

def load(): return pd.read_pickle(C.PRIVATE/'orders.pkl')

def prepare():
    C.directories();t,inv=read_tables();inv.to_csv(C.TABLES/'source_inventory.csv',index=False)
    x=features(t)
    orders=t['orders'].copy()
    orders['purchase_ts']=pd.to_datetime(orders.order_purchase_timestamp)
    orders['delivered_ts']=pd.to_datetime(orders.order_delivered_customer_date)
    orders['lead_time_days']=(orders.delivered_ts-orders.purchase_ts).dt.total_seconds()/86400
    ok=orders.order_status.eq('delivered')&orders.lead_time_days.notna()&orders.lead_time_days.ge(0)
    full=x.merge(orders[['order_id','delivered_ts','lead_time_days','order_status']],on='order_id',validate='one_to_one')
    all_valid=full[full.order_id.isin(orders.loc[ok,'order_id'])].copy()
    assert len(all_valid)==96470 and all_valid.order_id.is_unique
    # Verify the archived V2 fields independently of model code, when the local reference is supplied.
    reference=os.environ.get('REGRESSION_V2_REFERENCE')
    matches=None
    if reference:
        old=pd.read_csv(reference)
        both=all_valid.merge(old,on='order_id',suffixes=('_new','_old'),validate='one_to_one')
        assert len(both)==96470
        cols=['lead_time_days','max_distance_km','total_price','total_freight','freight_ratio',
              'item_count','seller_count','avg_product_weight_g','any_distance_missing','any_weight_missing','promised_lead_time_days']
        for c in cols: np.testing.assert_allclose(both[c+'_new'],both[c+'_old'],rtol=1e-10,atol=1e-8,equal_nan=True)
        matches=cols
    df=all_valid[(all_valid.purchase_ts>=C.START)&(all_valid.purchase_ts<C.TEST_END)].reset_index(drop=True)
    df.to_pickle(C.PRIVATE/'orders.pkl')
    all_valid.to_pickle(C.PRIVATE/'all_valid.pkl')
    raw=orders[(orders.purchase_ts>=C.START)&(orders.purchase_ts<C.TEST_END)]
    monthly=raw.assign(month=raw.purchase_ts.dt.strftime('%Y-%m'),valid=ok.loc[raw.index]).groupby('month').agg(
        source_orders=('order_id','size'),observed_valid_deliveries=('valid','sum')).reset_index()
    monthly['unresolved_or_excluded']=monthly.source_orders-monthly.observed_valid_deliveries
    monthly.to_csv(C.TABLES/'outcome_coverage_by_month.csv',index=False)
    flow=pd.DataFrame([
        ['Source orders',len(orders)],['Orders with items',len(x)],['Valid delivered orders (all dates)',len(all_valid)],
        ['Excluded outside Jan 2017-Aug 2018',len(all_valid)-len(df)],['Phase 3 modelling cohort',len(df)]],columns=['step','orders'])
    flow.to_csv(C.TABLES/'cohort_flow.csv',index=False)
    save_json(C.OUT/'preparation.json',{'source_orders':len(orders),'valid_all_dates':len(all_valid),
        'cohort_orders':len(df),'valid_over60_all_dates':int(all_valid.lead_time_days.gt(60).sum()),
        'reference_v2_fields_verified':matches,'input_sha256':digest(C.PRIVATE/'orders.pkl'),
        'timezone':'Source timezone-naive local timestamps retained; no invented UTC conversion.',
        'assumptions':['Archived quote, seller and product attributes assumed known at purchase.',
                       'Static geography reference has no historical availability log.',
                       'Evaluation conditional on observed completed deliveries; source snapshot has no authoritative extraction cutoff.']})
    return df,t
