
from config import ACCOUNT
from apis.oanda_api import OandaApi

oa = OandaApi()

summary = oa.getSummary(ACCOUNT)

account = {}
account['dt'] = summary['createdTime']
account['id'] = summary['id']
account['ccy'] = summary['currency']
account['balance'] = round(float(summary['balance']),2)
account['close_pnl'] = round(float(summary['resettablePL']),2)
account['NAV'] = round(float(summary['NAV']),2)
account['open_pnl'] = round(float(summary['unrealizedPL']),2)
account['commissions'] = round(float(summary['financing']),2)
account['margin_used'] = round(float(summary['marginUsed']),2)
account['open_position'] = round(float(summary['positionValue']),2)
account['leverage'] = round(account['open_position'] / account['NAV'],2)
