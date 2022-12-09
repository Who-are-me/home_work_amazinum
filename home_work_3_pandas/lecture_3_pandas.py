import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


# task 1
def answer_one():
    path_to_file_excel = os.path.join(os.getcwd(), "Energy_Indicators.xls")
    energy_df = pd.read_excel(path_to_file_excel)
    # don't use un need first two column
    energy_df = energy_df.drop(columns=energy_df.columns[0:2])
    # skip first 16 rows, because it's bad values
    energy_df = energy_df.iloc[17:244]

    # set unlimited for all showing dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # rename titles
    energy_df.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

    # converts to petajoule
    energy_df['Energy Supply'] = energy_df['Energy Supply'].apply(lambda x: np.NaN if x == '...' else x / 1_000_000)
    # converts all `...` to np.NaN
    energy_df['Energy Supply per Capita'] = energy_df['Energy Supply per Capita'].apply(
        lambda x: np.NaN if x == '...' else x)

    # Dict To Change Name Country
    dtcnc = {  # 'Republic of Korea': 'South Korea', but exists two Korea -_-
        'United States of America': 'United States',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'China, Hong Kong Special Administrative Region': 'Hong Kong'}

    energy_df['Country'] = energy_df['Country'].apply(lambda x: 'South Korea' if 'Republic of Korea' == str(x) else x)

    # rename name of countries by dtcnc
    for k, v in dtcnc.items():
        energy_df['Country'] = energy_df['Country'].apply(lambda x: v if k in str(x) else x)

    # delete number from name of country
    energy_df['Country'] = energy_df['Country'].apply(lambda x: ''.join(ch for ch in str(x) if not ch.isdigit()))

    # delete details in (some details)
    energy_df['Country'] = energy_df['Country'].apply(
        lambda x: str(x)[:str(x).find('(') - 1] if str(x).find('(') != -1 else x)

    path_to_file_csv = os.path.join(os.getcwd(), "world_bank.csv")
    # read csv file, without skiprows=3 i can't read this file (0v0)
    GDP = pd.read_csv(path_to_file_csv, skiprows=3)

    # drop last column, because bad value
    GDP = GDP.drop(columns=GDP.columns[-1:])

    # Dict To Rename Country Name
    dtrcn = {'Korea, Rep.': 'South Korea',
             'Iran, Islamic Rep.': 'Iran',
             'Hong Kong SAR, China': 'Hong Kong'}

    for k, v in dtrcn.items():
        GDP['Country Name'] = GDP['Country Name'].apply(lambda x: v if k == x else x)

    path_to_file_xlsx = os.path.join(os.getcwd(), 'scimagojr_country_rank_1996-2021.xlsx')
    ScimEn = pd.read_excel(path_to_file_xlsx)

    # ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', 2012', '2013', '2014', '2015']
    energy_and_GDP = pd.merge(energy_df, GDP.drop(columns=GDP.columns[1:50]), how='outer', left_on='Country',
                              right_on='Country Name')
    all_df = pd.merge(ScimEn.iloc[:15], energy_and_GDP.drop(columns=energy_and_GDP.columns[15:]), how='outer',
                      left_on='Country', right_on='Country')

    all_df = all_df.drop(columns=all_df.columns[1:3])
    all_df = all_df.drop(columns=['Country Name'])
    all_df = all_df[:15]

    # FOR CHECKING
    print('###')
    print(energy_df)
    print('###')
    print(GDP)
    print('###')
    print(ScimEn)
    print('###')
    print(all_df)

    return all_df


# average GDP of all country for 10 years
def answer_two():
    # set unlimited for all showing dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    sys.stderr.write("Data with names of regions of world!\n")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! data with names of regions of world !!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    path_to_file_csv = os.path.join(os.getcwd(), "world_bank.csv")
    # read csv file, without skiprows=3 i can't read this file (0v0)
    GDP = pd.read_csv(path_to_file_csv, skiprows=3)

    # drop last column, because bad value
    GDP = GDP.drop(columns=GDP.columns[-1:])

    # add empty column with 0.0
    GDP = GDP.assign(AverageGDP=0.0)

    for index, row in GDP.iterrows():
        div = 10 - pd.Series(row[-11:-1]).isnull().sum()
        GDP.at[index, 'AverageGDP'] = pd.Series(row[-11:-1]).sum() / (div if div != 0 else 1)  # minus count values equals NaN

    avgGDP = GDP.sort_values(by=['AverageGDP'], ascending=False)

    # but first 13 countries isn't country
    for ind in range(15):
        print("{:>45} => {}".format(avgGDP.iloc[ind][0], avgGDP.iloc[ind][-1]))

    # print(avgGDP)

    return avgGDP


# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
def answer_three():
    avgGDP = answer_two()
    avgGDP_list = []

    for it in range(10):
        avgGDP_list.append(avgGDP.iloc[5][-2 - it])

    avgGDP_list.reverse()

    x_ticks = [2012 + shift for shift in range(10)]

    # print(x_ticks)
    # print(avgGDP_list)

    fig, ax = plt.subplots()
    ax.plot(avgGDP_list)
    plt.title("Country name: " + avgGDP.iloc[5][0])
    plt.xlabel("Years")
    plt.ylabel("GDP in year")
    plt.xticks([x for x in range(10)], [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
    # bad view ####################################
    # plt.ticklabel_format(style='plain', axis='y')
    plt.show()

    # return changed in 10 years
    return avgGDP_list[-1] - avgGDP_list[0]


def answer_four():
    res = 0
    return res


def answer_five():
    res = 0
    return res


def answer_six():
    res = 0
    return res


def answer_seven():
    res = 0
    return res
