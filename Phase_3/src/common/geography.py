"""Zip-prefix coordinates and order route distances.

Moved from `Phase 2/.../Regression/src/data_preparation_audit.py` (geographic_reference,
compute_routes) so that both Phase 3 pipelines import one copy. Behaviour is unchanged
for the policy used in Phase 2 ("screened_median").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def geographic_reference(geo: pd.DataFrame, policy: str = "screened_median") -> pd.DataFrame:
    """Median coordinates per zip prefix. No outcomes are used.

    [-35, 6] latitude / [-75, -25] longitude is a generous screening envelope around
    Brazil's national extremes (Atlantic islands included); exact duplicate coordinates
    are dropped so that repeated reference rows do not add weight.
    """
    g = geo.copy()
    coordinates = ["geolocation_lat", "geolocation_lng"]
    if policy != "legacy_mean":
        inside = g.geolocation_lat.between(-35, 6) & g.geolocation_lng.between(-75, -25)
        g = g.loc[inside]
    if policy in ["screened_median", "screened_state_median"]:
        g = g.drop_duplicates(["geolocation_zip_code_prefix", "geolocation_state", *coordinates])
    keys = ["geolocation_zip_code_prefix"]
    if policy == "screened_state_median":
        keys.append("geolocation_state")
    grouped = g.groupby(keys)[coordinates]
    return grouped.mean() if policy in ["legacy_mean", "screened_mean"] else grouped.median()


def compute_routes(items, orders, customers, sellers, reference):
    """Haversine km for every order line (seller zip -> customer zip); per order the
    maximum distance and a flag for any line whose distance could not be computed."""
    route = (items[["order_id", "order_item_id", "seller_id"]]
             .merge(orders[["order_id", "customer_id"]], on="order_id", how="left", validate="many_to_one")
             .merge(customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
                    on="customer_id", how="left", validate="many_to_one")
             .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
                    on="seller_id", how="left", validate="many_to_one"))
    assert len(route) == len(items)
    for side in ["customer", "seller"]:
        route = route.merge(reference.rename(columns={"geolocation_lat": side + "_lat",
                                                      "geolocation_lng": side + "_lng"}),
                            left_on=side + "_zip_code_prefix", right_index=True,
                            how="left", validate="many_to_one")
    a, b, c, d = [np.radians(route[col]) for col in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]]
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    route["distance_km"] = (6371 * 2 * np.arcsin(np.sqrt(h.clip(0, 1)))).round(1)
    group = route.groupby("order_id").distance_km
    result = group.max().to_frame("max_distance_km")
    result["any_distance_missing"] = group.count().lt(group.size()).astype(int)
    return route, result
