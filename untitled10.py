# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 22:03:05 2023

@author: parkh
"""
import pandas as pd
q_time_data = "a.csv"
queue_time_table = pd.read_csv(q_time_data, index_col=(0))
print(queue_time_table)
for i in range(len(queue_time_table)):
    print(queue_time_table.iloc[i]["number"])
    print(queue_time_table.iloc[i]["duedate"])
    print(queue_time_table.iloc[i].name)
    
    