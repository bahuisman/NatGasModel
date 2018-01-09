
# coding: utf-8

# # Matching Market

# This simple model consists of a buyer, a supplier, and a market. 
# 
# The buyer represents a group of customers whose willingness to pay for a single unit of the good is captured by a vector of prices _wta_. You can initiate the buyer with a set_quantity function which randomly assigns the willingness to pay according to your specifications. You may ask for these willingness to pay quantities with a _getbid_ function. 
# 
# The supplier is similar, but instead the supplier is willing to be paid to sell a unit of technology. The supplier for instance may have non-zero variable costs that make them unwilling to produce the good unless they receive a specified price. Similarly the supplier has a  get_ask function which returns a list of desired prices. 
# 
# The willingness to pay or sell are set randomly using uniform random distributions. The resultant lists of bids are effectively a demand curve. Likewise the list of asks is effectively a supply curve. A more complex determination of bids and asks is possible, for instance using time of year to vary the quantities being demanded. 
# 
# ## New in version 15
# - Data read from file for import
# - storing data
# 
# ## Microeconomic Foundations
# 
# The market assumes the presence of an auctioneer which will create a _book_, which seeks to match the bids and the asks as much as possible. If the auctioneer is neutral, then it is incentive compatible for the buyer and the supplier to truthfully announce their bids and asks. The auctioneer will find a single price which clears as much of the market as possible. Clearing the market means that as many willing swaps happens as possible. You may ask the market object at what price the market clears with the get_clearing_price function. You may also ask the market how many units were exchanged with the get_units_cleared function.

# ## Agent-Based Objects
# 
# The following section presents three objects which can be used to make an agent-based model of an efficient, two-sided market. 

# In[5]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd
import numpy as np
import time
import datetime
import calendar
import json

# fix what is missing with the datetime/time/calendar package
def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year, month)[1])
    return datetime.date(year,month,day)


# ## classes buyers and sellers
# Below we are constructing the buyers and sellers in classes.

# In[7]:


# measure how long it takes to run the script
startit = time.time()
dtstartit = datetime.datetime.now()

class Seller():
    def __init__(self, name):
        self.name = name
        self.wta = []
        self.step = 0
        self.prod = 2000
        self.lb_price = 10
        self.lb_multiplier = 0
        self.ub_price = 20
        self.ub_multiplier = 0
        self.init_reserve = 500000
        self.reserve = 500000
        #multiple market idea, also 'go away from market'
        self.subscr_market = {}
        self.last_price = 0
        self.state_hist = {}
        self.cur_scenario = ''

    # the supplier has n quantities that they can sell
    # they may be willing to sell this quantity anywhere from a lower price of l
    # to a higher price of u
    def set_quantity(self):
        self.update_price()
        n = self.prod
        l = self.lb_price + self.lb_multiplier
        u = self.ub_price + self.ub_multiplier
        wta = []
        for i in range(n):
            p = rnd.uniform(l, u)
            wta.append(p)
        
        if len(wta) < self.reserve:
            self.wta = wta
        else:
            self.wta = wta[0:(self.reserve-1)]
            self.prod = self.reserve
        if len(self.wta) > 0:
            self.wta = sorted(self.wta, reverse=True)
        
    def get_name(self):
        return self.name

    def get_asks(self):
        return self.wta

    def extract(self, cur_extraction):
        if self.reserve > 0:
            self.reserve = self.reserve - cur_extraction
        else:
            self.prod = 0

    # production costs rise a 100% 
    def update_price(self):
        depletion = (self.init_reserve - self.reserve) / self.init_reserve
        self.ub_multiplier = int(self.ub_price * depletion)
        self.lb_multiplier = int(self.lb_price * depletion)

    # record every step into an dictionary, nog pythonic look into (vars)
    def book_keeping(self):
        self.state_hist[self.step] = self.__dict__
    
class Buyer():
    def __init__(self, name):
        self.name = name
        self.type = 0
        self.wtp = []
        self.step = 0
        self.base_demand = 0
        self.max_demand = 0
        self.lb_price = 10
        self.ub_price = 20
        self.subscr_market = {}
        self.state_hist = {}
        self.cur_scenario = ''
        self.consumed = 0

    # the supplier has n quantities that they can buy
    # they may be willing to sell this quantity anywhere from a lower price of l
    # to a higher price of u
    def set_quantity(self):
        self.update_price()
        n = int(self.consumption(self.step))
        l = self.lb_price
        u = self.ub_price
        wtp = []
        for i in range(n):
            p = rnd.uniform(l, u)
            wtp.append(p)
        self.wtp = sorted(wtp, reverse=False)
        
    # gets a little to obvious
    def get_name(self):
        return self.name
    
    # return list of willingness to pay
    def get_bids(self):
        return self.wtp
    
    def consumption(self, x):
        # make it initialise to seller
        b = self.base_demand
        m = self.max_demand
        y = b + m * (.5 * (1 + np.cos((x/6)*np.pi)))
        return(y)

    def consume(self, cur_extraction):
        self.consumed = self.consumed + cur_extraction
    
    # writes complete state to a dictionary, see if usefull
    def book_keeping(self):
        self.state_hist[self.step] = self.__dict__
        
    def update_price(self):
        if self.type == 1: #home
            if (self.step/12).is_integer():
                self.base_demand = home_savings[self.cur_scenario] * self.base_demand
                self.max_demand = home_savings[self.cur_scenario] * self.max_demand
        if self.type == 2: # elec
            if (self.step/12).is_integer():
                cur_elec_df = elec_space[self.cur_scenario]
                period_now = add_months(period_null, self.step)
                index_year = int(period_now.strftime('%Y'))
                demand = cur_elec_df[index_year]
                self.base_demand = demand * 7
                self.max_demand = demand * 12
        if self.type == 3: #indu
            if (self.step/12).is_integer():
                cur_df = economic_growth[self.cur_scenario]
                period_now = add_months(period_null, self.step)
                index_year = int(period_now.strftime('%Y'))
                growth = cur_df[index_year]
                self.base_demand = (1 + growth) * self.base_demand
                self.max_demand = (1 + growth) * self.max_demand
    

        
class storer():
    def __init__(self, name):
        self.name = name
        self.type = 0
        self.wtp = []
        self.wta = []
        self.step = 0
        self.base_demand = 0
        self.max_demand = 0
        self.lb_price = 10
        self.ub_price = 20
        self.subscr_market = {}
        self.state_hist = {}
        self.cur_scenario = ''        
        self.prod = 0
        self.reserve = 0
        self.capacity = 12000
        
    # gets a little to obvious
    def get_name(self):
        return self.name
    
    # return list of willingness to pay
    def get_bids(self):
        return self.wtp

    def get_asks(self):
        return self.wta
    
    def buying(self, x):
        # make it initialise to seller
        b = self.base_demand
        m = self.prod
        y = b + m * (.5 * (1 + np.sin((x/6)*np.pi)))
        return(y)

    def selling(self, x):
        # make it initialise to seller
        b = self.base_demand
        m = self.prod
        y = b + m * (.5 * (1 + np.cos((x/6)*np.pi)))
        return(y)
    
    def set_quantity(self):
        # what to sell
        #self.update_price()
        n = int(self.buying(self.step))
        if n > (self.capacity - self.reserve):
            n = self.capacity - self.reserve
            print(self.name, 'reserve full')
        l = self.lb_price
        u = self.ub_price
        wtp = []
        for i in range(n):
            p = rnd.uniform(l, u)
            wtp.append(p)
        self.wtp = sorted(wtp, reverse=False)
        
        n = int(self.selling(self.step))
        if n > self.reserve:
            n = self.reserve
            print(self.name, 'reserve empty')
        l = self.lb_price
        u = self.ub_price
        wta = []
        for i in range(n):
            p = rnd.uniform(l, u)
            wta.append(p)
        self.wta = sorted(wta, reverse=False)
        
    def extract(self, cur_extraction):
        if self.reserve > 0:
            self.reserve = self.reserve - cur_extraction
        else:
            #self.prod = 0
            
    def consume(self, cur_extraction):
        self.reserve = self.reserve + cur_extraction


# ## Construct the market
# For the market two classes are made. The market itself, which controls the buyers and the sellers, and the book. The market has a book where the results of the clearing procedure are stored.

# In[ ]:


# the book is an object of the market used for the clearing procedure
class Book():
    def __init__(self):
        self.ledger = pd.DataFrame(columns = ("role","name","price","cleared"))

    def set_asks(self,seller_list):
        # ask each seller their name
        # ask each seller their willingness
        # for each willingness append the data frame
        for seller in seller_list:
            seller_name = seller.get_name()
            seller_price = seller.get_asks()
            ## alternative way to construct the dataframe without numpy (twice as slow)
            #seller_df = []
            #for i in seller_price:
            #    seller_df.append(['seller',seller_name,i,'in process'])
            #temp_ledger = pd.DataFrame(seller_df, columns= ["role","name","price","cleared"])
            ar_role = np.full((1,len(seller_price)),'seller')
            ar_name = np.full((1,len(seller_price)),seller_name)
            ar_cleared = np.full((1,len(seller_price)),'in process')
            temp_ledger = pd.DataFrame([*ar_role,*ar_name,seller_price,*ar_cleared]).T
            temp_ledger.columns= ["role","name","price","cleared"]
            self.ledger = self.ledger.append(temp_ledger, ignore_index=True)

    def set_bids(self,buyer_list):
        # ask each seller their name
        # ask each seller their willingness
        # for each willingness append the data frame
        for buyer in buyer_list:
            buyer_name = buyer.get_name()
            buyer_price = buyer.get_bids()
            ## alternative way to construct the dataframe without numpy (twice as slow)
            #buyer_df = []
            #for i in buyer_price:
            #    buyer_df.append(['buyer',buyer_name,i,'in process'])
            #temp_ledger = pd.DataFrame(buyer_df, columns= ["role","name","price","cleared"])
            ar_role = np.full((1,len(buyer_price)),'buyer')
            ar_name = np.full((1,len(buyer_price)),buyer_name)
            ar_cleared = np.full((1,len(buyer_price)),'in process')
            temp_ledger = pd.DataFrame([*ar_role,*ar_name,buyer_price,*ar_cleared]).T
            temp_ledger.columns= ["role","name","price","cleared"]
            self.ledger = self.ledger.append(temp_ledger, ignore_index=True)
            
    def update_ledger(self,ledger):
        self.ledger = ledger
        
    def get_ledger(self):
        return self.ledger
    
    def clean_ledger(self):
        self.ledger = pd.DataFrame(columns = ("role","name","price","cleared"))

class Market():
    def __init__(self, name):
        self.name= name
        self.count = 0
        self.last_price = ''
        self.book = Book()
        self.b = []
        self.s = []
        self.buyer_list = []
        self.seller_list = []
        self.buyer_dict = {}
        self.seller_dict = {}
        self.ledger = ''
    
    # not called, if needed move to observer
    def book_keeping_all(self):
        for i in self.buyer_dict:
            self.buyer_dict[i].book_keeping()
        for i in self.seller_dict:
            self.seller_dict[i].book_keeping()
    
    def add_buyer(self,buyer):
        if buyer.subscr_market[self.name] == 1:
            self.buyer_list.append(buyer)
        
    def add_seller(self,seller):
        if seller.subscr_market[self.name] == 1:       
            self.seller_list.append(seller)
        
    def set_book(self):
        self.book.set_bids(self.buyer_list)
        self.book.set_asks(self.seller_list)
    
    def get_bids(self):
        # this is a data frame
        ledger = self.book.get_ledger()
        rows= ledger.loc[ledger['role'] == 'buyer']
        # this is a series
        prices=rows['price']
        # this is a list
        bids = prices.tolist()
        return bids
    
    def get_asks(self):
        # this is a data frame
        ledger = self.book.get_ledger()
        rows = ledger.loc[ledger['role'] == 'seller']
        # this is a series
        prices=rows['price']
        # this is a list
        asks = prices.tolist()
        return asks
    
    # return the price at which the market clears
    # this fails because there are more buyers then sellers
    
    def get_clearing_price(self):
        # buyer makes a bid starting with the buyer which wants it most
        b = self.get_bids()
        s = self.get_asks()
        # highest to lowest
        self.b=sorted(b, reverse=True)
        # lowest to highest
        self.s=sorted(s, reverse=False)
        
        # find out whether there are more buyers or sellers
        # then drop the excess buyers or sellers; they won't compete
        n = len(b)
        m = len(s)
        
        # there are more sellers than buyers
        # drop off the highest priced sellers 
        if (m > n):
            s = s[0:n]
            matcher = n
        # There are more buyers than sellers
        # drop off the lowest bidding buyers 
        else:
            b = b[0:m]
            matcher = m
        
        # It's possible that not all items sold actually clear the market here
        count = 0
        for i in range(matcher):
            if (self.b[i] > self.s[i]):
                count +=1
                self.last_price = self.b[i]
        
        # copy count to market object
        self.count = count
        return self.last_price
    
    # TODO: Annotate the ledger
    # this procedure takes up 80% of processing time
    def annotate_ledger(self,clearing_price):
        ledger = self.book.get_ledger()
        for index, row in ledger.iterrows():
            if (row['role'] == 'seller'):
                if (row['price'] < clearing_price):
                    ledger.loc[index,'cleared'] = 'True'
                else:
                    ledger.loc[index,'cleared'] = 'False'
            else:
                if (row['price'] > clearing_price):
                    ledger.loc[index,'cleared'] = 'True'
                else:
                    ledger.loc[index,'cleared'] = 'False'  
                    
        self.book.update_ledger(ledger)
    
    def get_units_cleared(self):
        return self.count
    
    def clean_ledger(self):
        self.ledger = ''
        self.book.clean_ledger()

    def run_it(self):
        self.pre_clearing_operation()
        self.clearing_operation()
        self.after_clearing_operation()
        
        # pre clearing procedures (eg. empty out the last run and start over)
        # clean ledger is kind of sloppy, rewrite functions to overide the ledger
    def pre_clearing_operation(self):
        self.clean_ledger()
        
    def clearing_operation(self):
        self.set_book()
        clearing_price = self.get_clearing_price()
        #start_clearing = datetime.datetime.now()      
        self.annotate_ledger(clearing_price)
        #print('timed operation:', datetime.datetime.now() - start_clearing)
        
        
        # After clearing operation (e.g. notify the agents about the outcome)
        # Notify sellers to extract resources
        # Removing the bids that cleared from the bid list the agents
    def after_clearing_operation(self):
        for agent in self.seller_list:
            name = agent.name
            cur_extract = len(self.book.ledger[(self.book.ledger['cleared'] == 'True') &
                              (self.book.ledger['name'] == name)])
            agent.extract(cur_extract)
            agent.last_price = self.last_price
            if cur_extract >0:
                agent.wta = agent.wta[:-cur_extract]
            #print(name, cur_extract)

        for agent in self.buyer_list:
            name = agent.name
            cur_extract = len(self.book.ledger[(self.book.ledger['cleared'] == 'True') &
                              (self.book.ledger['name'] == name)])
            if cur_extract >0:
                agent.wtp = agent.wtp[:-cur_extract]
            agent.consume(cur_extract)
            #print(name,cur_extract)

        # cleaning up the books
        #self.book_keeping_all()


# ## Observer
# The observer holds the clock and collects data. In this setup it tells the market another tick has past and it is time to act. The market will instruct the other agents. The observer initializes the model, thereby making real objects out of the classes defined above.

# In[ ]:


class Observer():
    def __init__(self, init_buyer, init_seller, timesteps, scenario):
        self.init_buyer = init_buyer
        self.init_seller = init_seller
        self.init_market = init_market
        self.init_storage = init_storage
        self.maxrun = timesteps
        self.cur_scenario = scenario
        self.hist_book = []
        self.buyer_dict = {}
        self.seller_dict = {}
        self.market_dict = {}
        self.storage_dict = {}
        self.timetick = 0
        self.gas_market = ''
        self.reserve = []
        self.all_data = {}

    def set_buyer(self, buyer_info):
        for name in buyer_info:
            self.buyer_dict[name] = Buyer('%s' % name)
            self.buyer_dict[name].base_demand = buyer_info[name]['b']
            self.buyer_dict[name].max_demand = buyer_info[name]['m']
            self.buyer_dict[name].lb_price = buyer_info[name]['lb_price']
            self.buyer_dict[name].ub_price = buyer_info[name]['ub_price']
            self.buyer_dict[name].type = buyer_info[name]['type']
            self.buyer_dict[name].cur_scenario = self.cur_scenario
            self.buyer_dict[name].subscr_market = dict.fromkeys(init_market,0)
            for market in buyer_info[name]['market']:
                self.buyer_dict[name].subscr_market[market] = 1

    def set_seller(self, seller_info):
        for name in seller_info:
            self.seller_dict[name] = Seller('%s' % name)
            self.seller_dict[name].prod = seller_info[name]['prod']
            self.seller_dict[name].lb_price = seller_info[name]['lb_price']
            self.seller_dict[name].ub_price = seller_info[name]['ub_price']
            self.seller_dict[name].reserve = seller_info[name]['reserve']
            self.seller_dict[name].init_reserve = seller_info[name]['reserve']
            self.seller_dict[name].cur_scenario = self.cur_scenario
            self.seller_dict[name].subscr_market = dict.fromkeys(init_market,0)
            for market in seller_info[name]['market']:
                self.seller_dict[name].subscr_market[market] = 1

    def set_storage(self, storage_info):
        for name in storage_info:
            self.storage_dict[name] = storer('%s' % name)
            self.storage_dict[name].prod = storage_info[name]['prod']
            self.storage_dict[name].reserve = storage_info[name]['reserve']
            self.storage_dict[name].capacity = storage_info[name]['capacity']
            self.storage_dict[name].cur_scenario = self.cur_scenario
            self.storage_dict[name].subscr_market = dict.fromkeys(init_market,0)
            for market in storage_info[name]['market']:
                self.storage_dict[name].subscr_market[market] = 1        
        
    def set_market(self, market_info):
        for name in market_info:
            self.market_dict[name] = Market('%s' % name)
        #self.gas_market = Market()
        #add suplliers and buyers to this market
            for supplier in self.seller_dict.values():
                self.market_dict[name].add_seller(supplier)
            for buyer in self.buyer_dict.values():
                self.market_dict[name].add_buyer(buyer)
            for storer in self.storage_dict.values():
                self.market_dict[name].add_buyer(storer)
                self.market_dict[name].add_seller(seller)
            self.market_dict[name].seller_dict = self.seller_dict
            self.market_dict[name].buyer_dict = self.buyer_dict
            
    def get_reserve(self):
        reserve = []
        for name in self.seller_dict:
            reserve.append(self.seller_dict[name].reserve)
        for name in self.storage_dict:
            reserve.append(self.storage_dict[name].reserve)
        return reserve
    
    def get_data(self):
        for name in self.seller_dict:
            self.all_data[name] = self.seller_dict[name].state_hist
        for name in self.buyer_dict:
            self.all_data[name] = self.buyer_dict[name].state_hist
    
    def update_buyer(self):
        for i in self.buyer_dict:
            self.buyer_dict[i].step += 1
            self.buyer_dict[i].set_quantity()
    
    def update_seller(self):
        for i in self.seller_dict:
            self.seller_dict[i].step += 1
            self.seller_dict[i].set_quantity()
            
    def update_storage(self):
        for i in self.storage_dict:
            self.storage_dict[i].step += 1
            self.storage_dict[i].set_quantity()
            
    def run_it(self):
        # Timing
        # time initialising
        startit_init = time.time()
        
        #initialise, setting up all the agents (firstrun not really needed anymore, since outside the loop)
        first_run = True
        if first_run:
            self.set_buyer(self.init_buyer)
            self.set_seller(self.init_seller)
            self.set_market(self.init_market)
            self.set_storage(self.init_storage)
            first_run=False
            
        # time init stop
        stopit_init = time.time() - startit_init
        print('%s : init' % stopit_init)
        
        for period in range(self.maxrun):
            # time the period
            startit_period = time.time()

            self.timetick += 1
            print('#######################################')
            period_now = add_months(period_null, self.timetick-1)
            print(period_now.strftime('%Y-%b'), self.cur_scenario)
            
            # update the buyers and sellers (timetick+ set Q)
            self.update_buyer()
            self.update_seller()
            
            # real action on the market
            for market in self.market_dict:
                self.market_dict[market].run_it()

            #self.gas_market.run_it()

            # data collection
            for name in self.market_dict:
                p_clearing = self.market_dict[name].last_price
                q_sold = self.market_dict[name].count
                self.reserve.append([period_now.strftime('%Y-%b'),*self.get_reserve()])
                self.hist_book.append([period_now.strftime('%Y-%b'), p_clearing, q_sold, name])
                
            # recording the step_info
            # since this operation can take quite a while, print after every operation
            period_time = time.time() - startit_period
            print('%.2f : seconds to clear period' % period_time)
            #self.hist_book.append([period_now.strftime('%Y-%b'), p_clearing, q_sold])


# ## Example Market
# 
# In the following code example we use the buyer and supplier objects to create a market. At the market a single price is announced which causes as many units of goods to be swapped as possible. The buyers and sellers stop trading when it is no longer in their own interest to continue. 

# In[ ]:


# import scenarios
inputfile = 'economic growth scenarios.xlsx'
economic_growth = pd.read_excel(inputfile, sheet='ec_growth', skiprows=1, index_col=0, header=0)

# demand for electricity import scenarios spaced by excel
elec_space = pd.read_excel(inputfile, sheetname='elec_space', index_col=0, header=0)

# gasdemand home (percentage increases)
home_savings = {'PACES': 1.01, 'TIDES': .99, 'CIRCLES': .97}


# In[ ]:


# make initialization dictionary
'''
init_buyer = {'elec_eu':{'b':400, 'm' : 673, 'lb_price': 10, 'ub_price' : 20, 'type' : 2, 'market' : ['eu']},
              'indu_eu':{'b':400, 'm':1171, 'lb_price': 10, 'ub_price' : 20, 'type' : 3, 'market' : ['eu']},
              'home_eu':{'b': 603, 'm': 3615, 'lb_price': 10, 'ub_price' : 20, 'type' : 1, 'market' : ['eu']},
              'elec_us':{'b':400, 'm' : 673, 'lb_price': 10, 'ub_price' : 20, 'type' : 2, 'market' : ['us']},
              'indu_us':{'b':400, 'm':1171, 'lb_price': 10, 'ub_price' : 20, 'type' : 3, 'market' : ['us']},
              'elec_as':{'b':400, 'm' : 673, 'lb_price': 10, 'ub_price' : 20, 'type' : 2, 'market' : ['as']},
              'indu_as':{'b':400, 'm':1171, 'lb_price': 10, 'ub_price' : 20, 'type' : 3, 'market' : ['as']}}

init_seller = {'NL' : {'prod': 2000, 'lb_price': 10, 'ub_price' : 20, 'reserve': 1044000, 'market' : ['eu']},
               'RU' : {'prod': 1000, 'lb_price': 15, 'ub_price' : 30, 'reserve': 32600000, 'market' : ['eu']},
               'NO': {'prod': 1000, 'lb_price': 13, 'ub_price' : 25, 'reserve': 1856000, 'market' : ['eu']},
               'US': {'prod': 1000, 'lb_price': 13, 'ub_price' : 25, 'reserve': 1856000, 'market' : ['us']},
               'QA': {'prod': 1000, 'lb_price': 13, 'ub_price' : 25, 'reserve': 1856000, 'market' : ['as','eu']}}
'''

init_storage = {'eu_stor' : {'capacity' : 12000, 'reserve' : 6000, 'prod': 2000, 'market': ['eu']}}

# constructing initialization data
# reading excel initialization data back
df_buyer = pd.read_excel('init_buyers_sellers.xlsx',orient='index',sheetname='buyers')
df_seller = pd.read_excel('init_buyers_sellers.xlsx',orient='index',sheetname='sellers')
# convert strings back to 
df_buyer['market'] = [eval(i) for i in df_buyer['market'].values]
df_seller['market'] = [eval(i) for i in df_seller['market'].values]
init_buyer = df_buyer.to_dict('index')
init_seller = df_seller.to_dict('index')

init_market = {'eu', 'us','as'}

# make a history book to record every timestep
hist_book = []

# set the starting time
period_null= datetime.date(2013,1,1)


# In[ ]:


# print to check if import worked out ok
for i in init_buyer:
    print(i, init_buyer[i])


# ## run the model
# To run the model we create the observer. The observer creates all the other objects and runs the model.

# In[ ]:


# create observer and run the model
# first data about buyers then sellers and then model ticks
years = 10
# timestep = 12
run_dict = {}
run_reserve = {}
run_data = {}
for i in ['CIRCLES', 'TIDES', 'PACES']:
    print(i)
    #cur_scenario = i
    obser1 = Observer(init_buyer, init_seller, years*12, i)
    obser1.run_it()
    #get the info from the observer
    run_dict[i] = obser1.hist_book
    run_reserve[i] = obser1.reserve
    run_data[i] = obser1.all_data


# In[ ]:


# check if objects match in library

print('last Q sold by different sellers')
for seller in obser1.seller_dict:
    print(obser1.seller_dict[seller].name, len(obser1.seller_dict[seller].wta))

print('\ntime step of different buyers')
for seller in obser1.buyer_dict:
    print(seller, obser1.buyer_dict[seller].step)

print('\nWhich sellers are active on what market')
for market in obser1.market_dict:
    print(market, [i.name for i in obser1.market_dict[market].seller_list])


# In[ ]:


# timeit
# runtime measurements to see performance impact of code changes
stopit = time.time()
dtstopit = datetime.datetime.now()

print('it took us %s seconds to get to this conclusion' % (stopit-startit))
print('in another notation (h:m:s) %s'% (dtstopit - dtstartit))


# ## Operations Research Formulation
# 
# The market can also be formulated as a very simple linear program or linear complementarity problem. It is clearer and easier to implement this market clearing mechanism with agents. One merit of the agent-based approach is that we don't need linear or linearizable supply and demand function. 
# 
# The auctioneer is effectively following a very simple linear program subject to constraints on units sold. The auctioneer is, in the primal model, maximizing the consumer utility received by customers, with respect to the price being paid, subject to a fixed supply curve. On the dual side the auctioneer is minimizing the cost of production for the supplier, with respect to quantity sold, subject to a fixed demand curve. It is the presumed neutrality of the auctioneer which justifies the honest statement of supply and demand. 
# 
# An alternative formulation is a linear complementarity problem. Here the presence of an optimal space of trades ensures that there is a Pareto optimal front of possible trades. The perfect opposition of interests in dividing the consumer and producer surplus means that this is a zero sum game. Furthermore the solution to this zero-sum game maximizes societal welfare and is therefore the Hicks optimal solution.
# 
# ## Next Steps
# 
# A possible addition of this model would be to have a weekly varying demand of customers, for instance caused by the use of natural gas as a heating agent. This would require the bids and asks to be time varying, and for the market to be run over successive time periods. A second addition would be to create transport costs, or enable intermediate goods to be produced. This would need a more elaborate market operator.  Another possible addition would be to add a profit maximizing broker. This may require adding belief, fictitious play, or message passing. 
# 
# The object-orientation of the models will probably need to be further rationalized. Right now the market requires very particular ordering of calls to function correctly. 

# ## Time of last run
# Time and date of the last run of this notebook file 

# In[ ]:


# print the time of last run
print('last run of this notebook:')
time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())


# ## Plotting scenario runs
# For the scenario runs we vary the external factors according to the scenarios. Real plotting is done in a seperate visualization file

# In[ ]:


plt.subplots()
for market in init_market:
    for i in run_dict:
        run_df = pd.DataFrame(run_dict[i])
        run_df = run_df[run_df[3]==market]
        run_df.set_index(0, inplace=True)
        run_df.index = pd.to_datetime(run_df.index)
        run_df.index.name = 'month'
        run_df.rename(columns={1: 'price', 2: 'quantity'}, inplace=True)
        run_df = run_df['price'].resample('A').mean().plot(label=i, title=market)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('€/MWh')
    plt.xlabel('Year')
    plt.show();


# ### saving data for later
# To keep this file as clear as possible and for efficiency we visualize the results in a separate file. To transfer the model run data we use the Json library (and possibly excel).

# In[ ]:


today = datetime.date.today().strftime('%Y%m%d')
outputexcel = '.\exceloutput\%srun.xlsx' %today
writer = pd.ExcelWriter(outputexcel)

def write_to_excel():
    for i in run_dict:
        run_df = pd.DataFrame(run_dict[i])
        run_df.set_index(0, inplace=True)
        run_df.index = pd.to_datetime(run_df.index)
        run_df.index.name = 'month'
        run_df.rename(columns={1: 'price', 2: 'quantity'}, inplace=True)
        run_df.to_excel(writer, sheet_name=i)
write_to_excel()


# In[ ]:


# Writing JSON data
# market data
data = run_dict
with open('marketdata.json', 'w') as f:
     json.dump(data, f)


# In[ ]:


# reserve data
data = run_reserve
with open('reservedata.json', 'w') as f:
     json.dump(data, f)


# # References
# 
# 
