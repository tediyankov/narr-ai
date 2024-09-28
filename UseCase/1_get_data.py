
## preliminaries =====================================================================

## libraries 
import os
import requests
import pandas as pd
from tqdm import tqdm

## functions =========================================================================

## Guardian scraper
def get_guardian_articles (from_date, to_date, search_terms):
    url = "https://content.guardianapis.com/search"
    articles = []
    for page in tqdm(range(1, 10), desc="Fetching articles"):
        params = {
            'from-date': from_date,
            'to-date': to_date,
            'page': page,
            'api-key': 'cd01b426-1a52-46fa-9ab0-09d56603c8c9',
            'q': ' OR '.join(search_terms),
            'show-fields': 'bodyText,headline,webPublicationDate',
            'page-size': 10
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'response' in data and 'results' in data['response']:
            articles.extend(data['response']['results'])
        else:
            break
        if data['response']['currentPage'] >= data['response']['pages']:
            break
    return articles 

## Df converter from Guardian article scraper output
def create_df_from_Guardian(articles):
    data = []
    for article in articles:
        date = article['webPublicationDate']
        title = article['webTitle']
        body = article['fields']['bodyText']
        data.append([date, title, body])
    df = pd.DataFrame(data, columns=['Date', 'Title', 'Body'])
    return df

def GuardianScraper (from_date, to_date, search_terms):
    articles = get_guardian_articles (from_date, to_date, search_terms)
    df = create_df_from_Guardian (articles)
    return df

## applying scraper ==================================================================

## parameters
search_terms = ['refugee', 'refugees', 'migrant', 'migrants', 'Refugee', 'Refugees', 'Migrant', 'Migrants', 'Migration', 'migration']

## running scraper
df_15_16 = GuardianScraper ('2015-01-01', '2016-01-01', search_terms)
df_16_17 = GuardianScraper ('2016-01-01', '2017-01-01', search_terms)
df_17_18 = GuardianScraper ('2017-01-01', '2018-01-01', search_terms)
df_18_19 = GuardianScraper ('2018-01-01', '2019-01-01', search_terms)
df_19_20 = GuardianScraper ('2019-01-01', '2020-01-01', search_terms)
df_20_21 = GuardianScraper ('2020-01-01', '2021-01-01', search_terms)

## concatenating data
guardian_df = pd.concat([df_15_16, df_16_17, df_17_18, df_18_19, df_19_20, df_20_21])

## saving data
guardian_df.to_csv('/data/exet5975/thesis/NarrAI/use_case/UseCase/new_guardian_corpus.csv', index = False)
