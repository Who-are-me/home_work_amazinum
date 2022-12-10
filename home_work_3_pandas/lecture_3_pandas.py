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

    # NEW TASK ###########################################################################

    path_to_file_xlsx = os.path.join(os.getcwd(), 'scimagojr_country_rank_1996-2021.xlsx')
    ScimEn = pd.read_excel(path_to_file_xlsx)

    # ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index',
    # 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', 2012',
    # '2013', '2014', '2015']
    energy_and_GDP = pd.merge(energy_df,
                              GDP.drop(columns=GDP.columns[1:50]),
                              how='outer',
                              left_on='Country',
                              right_on='Country Name')
    all_df = pd.merge(ScimEn.iloc[:15],
                      energy_and_GDP.drop(columns=energy_and_GDP.columns[15:]),
                      how='outer',
                      left_on='Country',
                      right_on='Country')

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
        print("{:>33} => {}".format(avgGDP.iloc[ind][0], avgGDP.iloc[ind][-1]))

    # FOR CHECKING
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
    default_x_ticks = [x for x in range(10)]

    fig, ax = plt.subplots()
    ax.plot(avgGDP_list)
    plt.title("Country name: " + avgGDP.iloc[5][0])
    plt.xlabel("Years")
    plt.ylabel("GDP in year")
    plt.xticks(default_x_ticks, x_ticks)

    # bad view ####################################
    # plt.ticklabel_format(style='plain', axis='y')
    # plt.figure(num='Home work by pandas (three answer)')

    # FOR CHECKING
    plt.show()

    # return changed in 10 years
    return avgGDP_list[-1] - avgGDP_list[0]


# Create a new column that is the ratio of Self-Citations to Total Citations.
# What is the maximum value for this new column, and what country has the highest ratio?
# This function should return a tuple with the name of the country and the ratio.
def answer_four():
    path_to_file_xlsx = os.path.join(os.getcwd(), 'scimagojr_country_rank_1996-2021.xlsx')
    df = pd.read_excel(path_to_file_xlsx)

    country = ""
    citations = 0

    # add empty column with 0.0
    df = df.assign(SelfCitationsToTotalCitations=0.0)

    for index, row in df.iterrows():
        div = df.iloc[index][5] if df.iloc[index][5] != 0 else 1
        df.at[index, 'SelfCitationsToTotalCitations'] = df.iloc[index][6] / div

    df = df.sort_values(by=['SelfCitationsToTotalCitations'], ascending=False)

    country = df.iloc[0][1]
    citations = df.iloc[0][-1]

    # FOR CHECKING
    # for index, row in df.iterrows():
    #     print(row.iloc[-1])
    # print(df.iloc[0])
    # print(country, citations)

    return (country, citations)


# Create a column that estimates the population using Energy Supply and Energy Supply per capita.
# What is the third most populous country according to this estimate?
# This function should return a single string value.
def answer_five():
    # set unlimited for all showing dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    path_to_file_excel = os.path.join(os.getcwd(), "Energy_Indicators.xls")
    energy_df = pd.read_excel(path_to_file_excel)
    # don't use un need first two column
    energy_df = energy_df.drop(columns=energy_df.columns[0:2])
    # skip first 16 rows, because it's bad values
    energy_df = energy_df.iloc[17:244]
    energy_df.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    # converts to petajoule
    energy_df['Energy Supply'] = energy_df['Energy Supply'].apply(lambda x: np.NaN if x == '...' else x / 1_000_000)
    # converts all `...` to np.NaN
    energy_df['Energy Supply per Capita'] = energy_df['Energy Supply per Capita'].apply(
        lambda x: np.NaN if x == '...' else x)
    # delete number from name of country
    energy_df['Country'] = energy_df['Country'].apply(lambda x: ''.join(ch for ch in str(x) if not ch.isdigit()))

    # add empty column with 0.0
    energy_df = energy_df.assign(PopulationByEnergySupply=0.0)

    for index, row in energy_df.iterrows():
        energy_df.at[index, 'PopulationByEnergySupply'] = row.iloc[1] * 1_000_000 / row.iloc[2] * 1_000_000

    energy_df = energy_df.sort_values(by=['PopulationByEnergySupply'], ascending=False)

    # FOR CHECKING
    # print(energy_df.iloc[212])
    # print(energy_df.iloc[79])
    # print(energy_df.iloc[0])
    # print(type(energy_df.iloc[79][2]))

    # answer is China :)
    return energy_df.iloc[0][0]


# Create a column that estimates the number of citable documents per person.
# What is the correlation between the number of citable documents per capita and the energy supply per capita?
# Use the .corr() method, (Pearson's correlation).
# This function should return a single number.
def answer_six():
    # set unlimited for all showing dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    path_to_file_xlsx = os.path.join(os.getcwd(), 'scimagojr_country_rank_1996-2021.xlsx')
    df = pd.read_excel(path_to_file_xlsx)

    path_to_file_excel = os.path.join(os.getcwd(), "Energy_Indicators.xls")
    energy_df = pd.read_excel(path_to_file_excel)

    # don't use un need first two column
    energy_df = energy_df.drop(columns=energy_df.columns[0:2])
    # skip first 16 rows, because it's bad values
    energy_df = energy_df.iloc[17:244]
    energy_df.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    # converts to petajoule
    energy_df['Energy Supply'] = energy_df['Energy Supply'].apply(lambda x: np.NaN if x == '...' else x / 1_000_000)
    # converts all `...` to np.NaN
    energy_df['Energy Supply per Capita'] = energy_df['Energy Supply per Capita'].apply(
        lambda x: np.NaN if x == '...' else x)
    # delete number from name of country
    energy_df['Country'] = energy_df['Country'].apply(lambda x: ''.join(ch for ch in str(x) if not ch.isdigit()))

    # add empty column with 0.0
    energy_df = energy_df.assign(PopulationByEnergySupply=0.0)

    for index, row in energy_df.iterrows():
        energy_df.at[index, 'PopulationByEnergySupply'] = row.iloc[1] * 1_000_000 / row.iloc[2] * 1_000_000

    # add empty column with 0.0
    df = df.assign(CitablePerPerson=0.0)

    merge_df = pd.merge(energy_df,
                        df,
                        how='outer',
                        left_on='Country',
                        right_on='Country')

    for index, row in merge_df.iterrows():
        div = row.iloc[9] if row.iloc[9] != 0 else 1
        merge_df.at[index, 'CitablePerPerson'] = row.iloc[4] / div

    list_of_energy_supply_per_person = []
    list_of_citable_per_person = []

    for index, row in merge_df.iterrows():
        list_of_energy_supply_per_person.append(row.iloc[2])
        list_of_citable_per_person.append(row.iloc[-1])

    # FOR CHECKING
    # print(merge_df.iloc[79])
    # print(list_of_energy_supply_per_person)
    # print(list_of_citable_per_person)

    s1 = pd.Series(list_of_energy_supply_per_person)
    s2 = pd.Series(list_of_citable_per_person)

    return s1.corr(s2, method='pearson')


# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays
# the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for
# the estimated population of each country.
# This function should return a DataFrame with index named Continent ['Asia', 'Australia', 'Europe', 'North America',
# 'South America'] and columns ['size', 'sum', 'mean', 'std']
def answer_seven():
    # Africa        => Africa/Middle East,  Africa
    # Asia          => Asiatic Region,      Middle East
    # Australia     => Pacific Region
    # Europe        => Western Europe,      Eastern Europe
    # North America => Northern America
    # South America => Latin America

    # set unlimited for all showing dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    path_to_file_xlsx = os.path.join(os.getcwd(), 'scimagojr_country_rank_1996-2021.xlsx')
    df = pd.read_excel(path_to_file_xlsx)

    path_to_file_excel = os.path.join(os.getcwd(), "Energy_Indicators.xls")
    energy_df = pd.read_excel(path_to_file_excel)

    # don't use un need first two column
    energy_df = energy_df.drop(columns=energy_df.columns[0:2])
    # skip first 16 rows, because it's bad values
    energy_df = energy_df.iloc[17:244]
    energy_df.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    # converts to petajoule
    energy_df['Energy Supply'] = energy_df['Energy Supply'].apply(lambda x: np.NaN if x == '...' else x / 1_000_000)
    # converts all `...` to np.NaN
    energy_df['Energy Supply per Capita'] = energy_df['Energy Supply per Capita'].apply(
        lambda x: np.NaN if x == '...' else x)
    # delete number from name of country
    energy_df['Country'] = energy_df['Country'].apply(lambda x: ''.join(ch for ch in str(x) if not ch.isdigit()))

    # add empty column with 0.0
    energy_df = energy_df.assign(PopulationByEnergySupply=0.0)

    for index, row in energy_df.iterrows():
        energy_df.at[index, 'PopulationByEnergySupply'] = row.iloc[1] * 1_000_000 / row.iloc[2] * 1_000_000

    # add empty column with 0.0
    df = df.assign(CitablePerPerson=0.0)

    merge_df = pd.merge(energy_df,
                        df,
                        how='outer',
                        left_on='Country',
                        right_on='Country')

    for index, row in merge_df.iterrows():
        div = row.iloc[9] if row.iloc[9] != 0 else 1
        merge_df.at[index, 'CitablePerPerson'] = row.iloc[4] / div

    merge_df["PopulationByEnergySupply"] = merge_df["PopulationByEnergySupply"].replace(np.nan, 0)

    # Africa        => Africa/Middle East,  Africa
    # Asia          => Asiatic Region,      Middle East
    # Australia     => Pacific Region
    # Europe        => Western Europe,      Eastern Europe
    # North America => Northern America
    # South America => Latin America

    list_of_size = [0, 0, 0, 0, 0, 0]
    list_of_sum = [0, 0, 0, 0, 0, 0]
    list_of_mean = [[],[],[],[],[],[]]
    list_of_std = [[],[],[],[],[],[]]

    def set_data_to_lists(ind, data):
        list_of_size[ind] += 1
        list_of_sum[ind] += data
        list_of_mean[ind].append(data)
        list_of_std[ind].append(data)

    for index, row in merge_df.iterrows():
        item = row.iloc[6]
        population = row.iloc[4]

        if item == 'Africa' or item == 'Africa/Middle East':
            set_data_to_lists(0, population)
        elif item == 'Asiatic Region' or item == 'Middle East':
            set_data_to_lists(1, population)
        elif item == 'Pacific Region':
            set_data_to_lists(2, population)
        elif item == 'Western Europe' or item == 'Eastern Europe':
            set_data_to_lists(3, population)
        elif item == 'Northern America':
            set_data_to_lists(4, population)
        elif item == 'Latin America':
            set_data_to_lists(5, population)

    for index in range(6):
        list_of_mean[index] = pd.Series(list_of_mean[index]).mean()
        list_of_std[index] = pd.Series(list_of_std[index]).mean()

    list_for_df = [list_of_size, list_of_sum, list_of_mean, list_of_std]

    res_df = pd.DataFrame(list_for_df,
                          index=['Size', 'Sum', 'Mean', 'Std'],
                          columns=['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America'])

    # FOR CHECKING
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # print(merge_df.iloc[79])
    # print(list_of_energy_supply_per_person)
    # print(list_of_citable_per_person)
    # print(res_df)

    return res_df
