import pandas as pd

class Portfolio:
    """ Data structure for a stock trading portfolio. """
    def __init__(self, init_balance):
        self._init_balance = init_balance
        self.balance = init_balance
        self.shares_owned = 0
        self.cost_basis = 0
    
    def get_balance(self) -> float:
        return self.balance
    
    def get_shares_owned(self) -> int:
        return self.shares_owned

    def get_net_worth(self, share_price) -> float:
        """ Returns net worth of Portfolio where shares sell at share_price """
        return self.balance + (self.shares_owned * share_price)

    def get_profit(self, share_price) -> float:
        """ Returns profit from beginning of environment """
        return self.get_net_worth(share_price) - self._init_balance

    def get_cost_basis(self) -> float:
        return self.cost_basis

    def buy(self, num_shares, share_price):
        """ Buy num_shares of shares at share_price each """
        assert num_shares * share_price <= self.balance

        prev_cost = self.get_owned_share_value()
        additional_cost = num_shares * share_price

        self.balance -= additional_cost
        self.cost_basis = (prev_cost + additional_cost) / (self.shares_owned + num_shares)
        self.shares_owned += num_shares

    def sell(self, num_shares, share_price):
        """ Sell num_shares of shares at share_price each """
        assert num_shares <= self.shares_owned

        self.balance += num_shares * share_price
        self.shares_owned -= num_shares

        if self.shares_owned == 0:
            self.cost_basis = 0

    def get_owned_share_value(self) -> float:
        return self.cost_basis * self.shares_owned

    def as_df(self):
        """ Returns the portfolio as a pandas DataFrame """
        data = {
            'balance': [self.balance],
            'shares_owned': [self.shares_owned],
            'cost_basis': [self.cost_basis],
            'net_worth': [self.get_net_worth(self.cost_basis)],
        }
        return pd.DataFrame(data)