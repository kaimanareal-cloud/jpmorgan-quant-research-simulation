"""
Natural Gas Storage Contract Pricer
------------------------------------
Value = Revenue from withdrawals
      - Cost of gas purchases
      - Storage flat fee  (monthly, first injection -> last withdrawal)
      - Injection cost    (per MMBtu injected)
      - Withdrawal cost   (per MMBtu withdrawn)
      - Transport cost    (per trip: each injection date + each withdrawal date = 1 trip each)
"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_prices,
    withdrawal_prices,
    injection_volumes,
    withdrawal_volumes,
    injection_rate,
    withdrawal_rate,
    max_storage_volume,
    storage_cost_per_month,
    injection_cost_per_mmbtu=0.0,
    withdrawal_cost_per_mmbtu=0.0,
    transport_cost_per_trip=0.0,
    verbose=True,
):
    """
    Price a natural gas storage contract.

    Parameters
    ----------
    injection_dates         : list of str or datetime — dates gas is purchased & injected
    withdrawal_dates        : list of str or datetime — dates gas is sold & withdrawn
    injection_prices        : list of float — purchase price ($/MMBtu) on each injection date
    withdrawal_prices       : list of float — sale price ($/MMBtu) on each withdrawal date
    injection_volumes       : list of float — MMBtu to inject on each injection date
    withdrawal_volumes      : list of float — MMBtu to withdraw on each withdrawal date
    injection_rate          : float — max MMBtu that can be injected on a single date
    withdrawal_rate         : float — max MMBtu that can be withdrawn on a single date
    max_storage_volume      : float — maximum MMBtu the facility can hold at any time
    storage_cost_per_month  : float — flat $/month fee from first injection to last withdrawal
    injection_cost_per_mmbtu: float — facility fee per MMBtu injected (default 0)
    withdrawal_cost_per_mmbtu: float — facility fee per MMBtu withdrawn (default 0)
    transport_cost_per_trip : float — transport cost per trip; each injection or withdrawal
                                      date counts as one trip (default 0)

    Returns
    -------
    dict with keys:
        value                  — net contract value ($)
        revenue                — total gas sale revenue ($)
        purchase_cost          — total gas purchase cost ($)
        storage_cost           — total storage fee ($)
        injection_cost         — total injection facility fee ($)
        withdrawal_cost        — total withdrawal facility fee ($)
        transport_cost         — total transport cost ($)
        storage_months         — months charged for storage
        final_inventory        — MMBtu remaining in storage at end (should be 0)
    """
    # ── Parse dates ───────────────────────────────────────────────────────────
    inj_dates  = [pd.to_datetime(d) for d in injection_dates]
    with_dates = [pd.to_datetime(d) for d in withdrawal_dates]

    n_inj  = len(inj_dates)
    n_with = len(with_dates)

    # ── Input length checks ───────────────────────────────────────────────────
    assert len(injection_prices)   == n_inj,  "injection_prices length must match injection_dates"
    assert len(withdrawal_prices)  == n_with, "withdrawal_prices length must match withdrawal_dates"
    assert len(injection_volumes)  == n_inj,  "injection_volumes length must match injection_dates"
    assert len(withdrawal_volumes) == n_with, "withdrawal_volumes length must match withdrawal_dates"

    # ── Rate constraint checks ────────────────────────────────────────────────
    for i, vol in enumerate(injection_volumes):
        if vol > injection_rate:
            raise ValueError(
                f"Injection volume {vol} MMBtu on {inj_dates[i].date()} "
                f"exceeds injection rate limit of {injection_rate} MMBtu."
            )
    for i, vol in enumerate(withdrawal_volumes):
        if vol > withdrawal_rate:
            raise ValueError(
                f"Withdrawal volume {vol} MMBtu on {with_dates[i].date()} "
                f"exceeds withdrawal rate limit of {withdrawal_rate} MMBtu."
            )

    # ── Inventory simulation (chronological) ─────────────────────────────────
    events = []
    for d, vol, price in zip(inj_dates, injection_volumes, injection_prices):
        events.append({"date": d, "type": "inject", "volume": vol, "price": price})
    for d, vol, price in zip(with_dates, withdrawal_volumes, withdrawal_prices):
        events.append({"date": d, "type": "withdraw", "volume": vol, "price": price})

    events.sort(key=lambda x: x["date"])

    inventory = 0.0
    for ev in events:
        if ev["type"] == "inject":
            inventory += ev["volume"]
            if inventory > max_storage_volume:
                raise ValueError(
                    f"Injecting {ev['volume']} MMBtu on {ev['date'].date()} pushes inventory "
                    f"to {inventory:.0f} MMBtu, exceeding max capacity of {max_storage_volume} MMBtu."
                )
        else:
            inventory -= ev["volume"]
            if inventory < 0:
                raise ValueError(
                    f"Withdrawing {ev['volume']} MMBtu on {ev['date'].date()} exceeds "
                    f"available inventory. Storage would go negative ({inventory:.0f} MMBtu)."
                )

    final_inventory = inventory

    # ── Storage duration (first injection → last withdrawal) ──────────────────
    contract_start = min(inj_dates)
    contract_end   = max(with_dates)

    # Fractional months using relativedelta for accuracy
    delta = relativedelta(contract_end, contract_start)
    storage_months = delta.years * 12 + delta.months + delta.days / 30.0

    # ── Cash flow calculations ────────────────────────────────────────────────
    revenue          = sum(p * v for p, v in zip(withdrawal_prices, withdrawal_volumes))
    purchase_cost    = sum(p * v for p, v in zip(injection_prices,  injection_volumes))
    storage_cost     = storage_cost_per_month * storage_months
    inj_facility_fee = injection_cost_per_mmbtu  * sum(injection_volumes)
    with_facility_fee= withdrawal_cost_per_mmbtu * sum(withdrawal_volumes)
    transport_cost   = transport_cost_per_trip * (n_inj + n_with)

    net_value = (
        revenue
        - purchase_cost
        - storage_cost
        - inj_facility_fee
        - with_facility_fee
        - transport_cost
    )

    result = {
        "value":             round(net_value, 2),
        "revenue":           round(revenue, 2),
        "purchase_cost":     round(purchase_cost, 2),
        "storage_cost":      round(storage_cost, 2),
        "injection_cost":    round(inj_facility_fee, 2),
        "withdrawal_cost":   round(with_facility_fee, 2),
        "transport_cost":    round(transport_cost, 2),
        "storage_months":    round(storage_months, 4),
        "final_inventory":   round(final_inventory, 2),
    }

    if verbose:
        _print_summary(result, contract_start, contract_end)

    return result


def _print_summary(r, start, end):
    width = 48
    print("=" * width)
    print("  Gas Storage Contract Valuation")
    print(f"  {start.date()}  ->  {end.date()}  ({r['storage_months']:.2f} months)")
    print("=" * width)
    print(f"  {'Revenue (withdrawals)':<30} ${r['revenue']:>12,.2f}")
    print(f"  {'Purchase cost (injections)':<30} ${r['purchase_cost']:>12,.2f}")
    print(f"  {'Storage fee':<30} ${r['storage_cost']:>12,.2f}")
    print(f"  {'Injection facility fee':<30} ${r['injection_cost']:>12,.2f}")
    print(f"  {'Withdrawal facility fee':<30} ${r['withdrawal_cost']:>12,.2f}")
    print(f"  {'Transport cost':<30} ${r['transport_cost']:>12,.2f}")
    print("-" * width)
    print(f"  {'NET CONTRACT VALUE':<30} ${r['value']:>12,.2f}")
    print("=" * width)
    if r["final_inventory"] != 0:
        print(f"  WARNING: {r['final_inventory']:.0f} MMBtu remain in storage at contract end.")


# ── Test Cases ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n── TEST 1: Textbook example from problem description ──")
    print("Buy 1M MMBtu at $2, sell at $3, store 4 months,")
    print("$100K/month storage, $10K/1M MMBtu inj+with fee, $50K transport per trip")
    price_storage_contract(
        injection_dates          = ["2021-06-30"],
        withdrawal_dates         = ["2021-10-31"],
        injection_prices         = [2.0],
        withdrawal_prices        = [3.0],
        injection_volumes        = [1_000_000],
        withdrawal_volumes       = [1_000_000],
        injection_rate           = 1_000_000,
        withdrawal_rate          = 1_000_000,
        max_storage_volume       = 1_000_000,
        storage_cost_per_month   = 100_000,
        injection_cost_per_mmbtu = 10_000 / 1_000_000,
        withdrawal_cost_per_mmbtu= 10_000 / 1_000_000,
        transport_cost_per_trip  = 50_000,
    )
    # Expected: (3-2)*1e6 - 400K storage - 10K inj - 10K with - 100K transport = $480,000

    print("\n── TEST 2: Seasonal trade using get_price() prices ──")
    print("Buy cheap in summer 2023, sell at winter premium 2023/24")
    from nat_gas_pricer import get_price
    summer_price = get_price("2023-06-30")
    winter_price = get_price("2024-01-31")
    print(f"  Summer buy price : ${summer_price:.4f}/MMBtu")
    print(f"  Winter sell price: ${winter_price:.4f}/MMBtu")
    price_storage_contract(
        injection_dates          = ["2023-06-30"],
        withdrawal_dates         = ["2024-01-31"],
        injection_prices         = [summer_price],
        withdrawal_prices        = [winter_price],
        injection_volumes        = [500_000],
        withdrawal_volumes       = [500_000],
        injection_rate           = 500_000,
        withdrawal_rate          = 500_000,
        max_storage_volume       = 500_000,
        storage_cost_per_month   = 50_000,
        injection_cost_per_mmbtu = 0.01,
        withdrawal_cost_per_mmbtu= 0.01,
        transport_cost_per_trip  = 0,
    )

    print("\n── TEST 3: Multiple injection & withdrawal dates ──")
    print("Inject across two summer months, withdraw over two winter months")
    from nat_gas_pricer import get_price
    price_storage_contract(
        injection_dates          = ["2022-05-31", "2022-06-30"],
        withdrawal_dates         = ["2022-11-30", "2022-12-31"],
        injection_prices         = [get_price("2022-05-31"), get_price("2022-06-30")],
        withdrawal_prices        = [get_price("2022-11-30"), get_price("2022-12-31")],
        injection_volumes        = [300_000, 200_000],
        withdrawal_volumes       = [250_000, 250_000],
        injection_rate           = 300_000,
        withdrawal_rate          = 250_000,
        max_storage_volume       = 500_000,
        storage_cost_per_month   = 25_000,
        injection_cost_per_mmbtu = 0.005,
        withdrawal_cost_per_mmbtu= 0.005,
        transport_cost_per_trip  = 10_000,
    )

    print("\n── TEST 4: Contract with no profit (costs wipe out spread) ──")
    price_storage_contract(
        injection_dates          = ["2023-07-31"],
        withdrawal_dates         = ["2023-12-31"],
        injection_prices         = [get_price("2023-07-31")],
        withdrawal_prices        = [get_price("2023-12-31")],
        injection_volumes        = [100_000],
        withdrawal_volumes       = [100_000],
        injection_rate           = 100_000,
        withdrawal_rate          = 100_000,
        max_storage_volume       = 100_000,
        storage_cost_per_month   = 200_000,
        injection_cost_per_mmbtu = 0.01,
        withdrawal_cost_per_mmbtu= 0.01,
        transport_cost_per_trip  = 0,
    )

    print("\n── TEST 5: Constraint violation — should raise error ──")
    try:
        price_storage_contract(
            injection_dates          = ["2023-06-30"],
            withdrawal_dates         = ["2023-12-31"],
            injection_prices         = [10.0],
            withdrawal_prices        = [12.0],
            injection_volumes        = [600_000],   # exceeds rate limit below
            withdrawal_volumes       = [600_000],
            injection_rate           = 500_000,     # rate cap
            withdrawal_rate          = 600_000,
            max_storage_volume       = 1_000_000,
            storage_cost_per_month   = 10_000,
            verbose=True,
        )
    except ValueError as e:
        print(f"  Caught expected error: {e}")
