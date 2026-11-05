"""Inspect course-style data contracts without changing the modelling cohort.

The GX teaching lab's declarative checks are implemented with pandas here; GX
itself is not a dependency. Row identifiers are written only under ignored runs/.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

FILES = {
    'orders': 'olist_orders_dataset.csv',
    'items': 'olist_order_items_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'geo': 'olist_geolocation_dataset.csv',
    'translation': 'product_category_name_translation.csv',
}
STATUSES = {'created', 'approved', 'invoiced', 'processing', 'shipped',
            'delivered', 'canceled', 'unavailable'}
ARCHIVE_SHA = '90ee50730a9e799aa2d3c7b1758fef680cbbc366f61a287870144796a37156d4'


def audit_course_archive(archive, destination):
    """Return aggregate evidence and retain all unexpected rows for inspection."""
    archive, destination = Path(archive), Path(destination)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('Use the unchanged course ZIP; reconcile a different hash first.')
    destination.mkdir(parents=True, exist_ok=False)
    tables, inventory = {}, []
    with zipfile.ZipFile(archive) as source:
        for table, filename in FILES.items():
            content = source.read('Olist_CSV/' + filename)
            tables[table] = pd.read_csv(BytesIO(content))
            inventory.append({'table': table, 'rows': len(tables[table]),
                              'sha256': hashlib.sha256(content).hexdigest()})
    records, unexpected = [], []

    def check(name, table, mask, action, hard=True):
        frame = tables[table]
        bad = pd.Series(mask, index=frame.index).fillna(True).astype(bool)
        count = int(bad.sum())
        records.append({'check': name, 'table': table, 'rows_checked': len(frame),
                        'unexpected_rows': count, 'gate': 'STOP' if hard else 'REVIEW',
                        'result': 'PASS' if count == 0 else ('FAIL' if hard else 'FLAG'),
                        'action': action})
        for index in frame.index[bad]:
            row = frame.loc[index]
            identifier = next((str(row[c]) for c in ['order_id', 'product_id', 'customer_id',
                                                    'seller_id', 'geolocation_zip_code_prefix']
                               if c in frame.columns), str(index))
            unexpected.append({'check': name, 'table': table, 'source_row': int(index),
                               'record_key': identifier})

    for table, key in [('orders', 'order_id'), ('customers', 'customer_id'),
                       ('sellers', 'seller_id'), ('products', 'product_id'),
                       ('translation', 'product_category_name')]:
        values = tables[table][key]
        check(key + ' present and unique', table,
              values.isna() | values.duplicated(keep=False), 'Stop before joins.')
    items, orders = tables['items'], tables['orders']
    check('Item composite key present and unique', 'items',
          items[['order_id', 'order_item_id']].isna().any(axis=1)
          | items.duplicated(['order_id', 'order_item_id'], keep=False), 'Stop before aggregation.')
    for column, target, key in [('order_id', 'orders', 'order_id'),
                                ('product_id', 'products', 'product_id'),
                                ('seller_id', 'sellers', 'seller_id')]:
        check('Item ' + column + ' foreign key exists', 'items',
              ~items[column].isin(tables[target][key]), 'Stop before joins.')
    check('Order customer foreign key exists', 'orders',
          ~orders.customer_id.isin(tables['customers'].customer_id), 'Stop before joins.')
    purchase = pd.to_datetime(orders.order_purchase_timestamp, errors='coerce')
    receipt = pd.to_datetime(orders.order_delivered_customer_date, errors='coerce')
    check('Purchase timestamp present and parseable', 'orders', purchase.isna(), 'Stop before target creation.')
    check('Order status in allowed values', 'orders', ~orders.order_status.isin(STATUSES), 'Stop and reconcile.')
    check('Order count within teaching sanity range', 'orders',
          np.full(len(orders), not 1000 <= len(orders) <= 500000), 'Stop and reconcile the source.')
    check('Item price strictly positive', 'items', items.price.isna() | items.price.le(0),
          'Review unexpected values; do not silently change the frozen cohort.', hard=False)
    check('Freight non-negative (zero allowed)', 'items',
          items.freight_value.isna() | items.freight_value.lt(0),
          'Review unexpected values; zero freight is allowed.', hard=False)
    check('Delivered receipt timestamp observable', 'orders',
          orders.order_status.eq('delivered') & receipt.isna(),
          'Exclude the 8 unobservable targets, not all missing raw receipts.', hard=False)
    check('Delivered purchase-to-receipt target non-negative', 'orders',
          orders.order_status.eq('delivered') & receipt.notna() & receipt.lt(purchase),
          'Exclude invalid targets; retain valid long durations.', hard=False)
    geo = tables['geo']
    check('Coordinates inside conservative Brazil envelope', 'geo',
          ~geo.geolocation_lat.between(-35, 6) | ~geo.geolocation_lng.between(-75, -25),
          'Screen these 29 coordinate rows, deduplicate pairs, then use postal-prefix medians.', hard=False)
    check('Product weight observable and positive', 'products',
          tables['products'].product_weight_g.isna() | tables['products'].product_weight_g.le(0),
          'Keep orders; mark incomplete weight and fit imputation inside the training partition.', hard=False)
    contracts = pd.DataFrame(records)
    contracts.to_csv(destination / 'data_contracts.csv', index=False)
    pd.DataFrame(inventory).to_csv(destination / 'source_inventory.csv', index=False)
    pd.DataFrame(unexpected, columns=['check', 'table', 'source_row', 'record_key']).to_csv(
        destination / 'unexpected_rows.csv', index=False)
    summary = {'archive_sha256': ARCHIVE_SHA, 'checks': len(records),
               'hard_checks_passed': not contracts.result.eq('FAIL').any(),
               'rule_violations_are_not_a_unique_order_count': True,
               'GX_library_used': False, 'cohort_or_model_changed': False}
    (destination / 'audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['hard_checks_passed']:
        raise ValueError('Structural data gate failed; inspect runs/ unexpected_rows.csv before modelling.')
    return contracts, pd.DataFrame(inventory), summary


def audit_random_frames(train, test):
    """Check the actual random frames, rather than the legacy prepared split."""
    train_ids, test_ids = set(train.order_id), set(test.order_id)
    assert len(train) == 64634 and len(test) == 31836
    assert train.order_id.is_unique and test.order_id.is_unique
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == 96470
    assert train.oof_fold.between(1, 5).all()
    assert (train.lead_time_days.ge(0) & np.isfinite(train.lead_time_days)).all()
    assert (test.lead_time_days.ge(0) & np.isfinite(test.lead_time_days)).all()
    assert int(train.lead_time_days.gt(60).sum() + test.lead_time_days.gt(60).sum()) == 306
    rows = []
    for fold in range(1, 6):
        validation = train.loc[train.oof_fold.eq(fold)]
        fitting = train.loc[train.oof_fold.ne(fold)]
        assert set(fitting.order_id).isdisjoint(validation.order_id)
        assert set(validation.order_id).isdisjoint(test_ids)
        rows.append({'fold': fold, 'fit_orders': len(fitting),
                     'validation_orders': len(validation), 'test_orders_in_CV': 0})
    assert sum(row['validation_orders'] for row in rows) == len(train)
    return pd.DataFrame(rows)
