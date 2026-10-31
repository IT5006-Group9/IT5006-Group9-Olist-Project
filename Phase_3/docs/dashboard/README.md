# Dashboard screenshots

Captured from the [deployed Phase 3 application](https://it5006-group9-olist-project-c5hmxth6jt543wszxqoaqd.streamlit.app/)
after online browser checks on 31 October 2026.
The model demonstration executes the fitted regressor, then optionally reveals
the held-out historical delivery outcome. The screenshots may be reused in the
presentation and deployment section of the report.

## Overview

![Olist Operations Studio overview](overview.png)

## Shared filters and mobile layout update

The updated Overview and Marketplace explorer retain the same selection across
page changes. Both offer Reset all filters, and Overview shows extra conditions
set in the explorer. At 360 px the sidebar starts collapsed and the four summary
metrics use a two-column layout.

![Updated mobile overview and shared filters](mobile_overview.jpg)

## Historical example navigation

Previous, Next and Random browse within the current month and destination.
Changing an order or leaving the prediction page clears its forecast and revealed
outcome. The screenshot shows a real prediction with the outcome still hidden.

![Updated historical example navigation and forecast](prediction_navigation.jpg)

## Real prediction and revealed historical outcome

![Delivery prediction and outcome comparison](prediction.png)

## Classification timing audit notice

Captured after the audit update and Cloud reboot. The archived scores remain visible with a notice that strict feature and label availability require a new assessment. The regression evidence page also displays the saved selection-maturity and duration-tail diagnostics.

![Classification timing audit notice](classification_timing.jpg)

See the [dashboard guide](../../../dashboard/README.md) for setup, prediction
semantics, evaluation sources and limitations.
