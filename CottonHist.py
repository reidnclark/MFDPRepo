# 1) CONVERT TIMEZONE TO PST AND CALGARY TIME
# - datetime_beginning_utc, datetime_beginning_ept
# 2) GRID LOCATION (e.g. Generator, Load, Transmission Grid Part)
# - pnode_id is the identifier, and pnode_name is readable version
# 3) VOLTAGE = voltage at node. Higher voltage indicates likely...
# ...a long-distance trasnmission node, and lower is maybe a...
# ...local distribution

import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)


def extractData (csv: str, 
                parameters: list[str]) -> pd.DataFrame:
    data = pd.read_csv(csv)[parameters]
    return data


data = extractData('rt_hrl_lmps.csv', # File name (csv)
                    ['datetime_beginning_utc', # Hour of instance
                    'pnode_id', # Unique identifier (check if related to zone???)
                    'pnode_name', # Node name (in-zone identifier)
                    'voltage',
                    'type', # Type of facilitation (e.g. Load, Generation)...
                    # ...LOAD = consumption point (e.g. City, Residential)...
                    # ...GEN = generation point (e.g.)
                    'zone', # Zone of node (zone / regional identifier)
                    'system_energy_price_rt', # System price
                    'total_lmp_rt', # Total marginal price (locational-marginal)...
                    # ...This includes congestion and marginal loss effects.
                    'congestion_price_rt',
                    'marginal_loss_price_rt'
                    ])

idxs = data.columns

print(data.head())